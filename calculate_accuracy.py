import networkx as nx
from networkx.algorithms import community
import community.community_louvain as community_louvain
import itertools
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk, ImageDraw  # Requires: pip install pillow
import os


def read_results(results_path):
    open(results_path, "r")  # Ensure file exists
    results = {}
    with open(results_path, "r") as f:
        key = None
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue  # Skip malformed lines
            image, reid = parts

            if reid == "1.0":
                key = image
                continue
            else:
                if key in results:
                    results[key].append((image, reid))
                else:
                    results[key] = [(image, reid)]
    return results


def read_train(train_path):
    open(train_path, "r")  # Ensure file exists
    train_data = {}
    with open(train_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            reid = parts[1]
            image = parts[3]

            if reid != "":
                if reid in train_data:
                    train_data[reid].append(image)
                else:
                    train_data[reid] = [image]
            else:
                # Treat empty reid as singleton with unique label (using image path)
                if image in train_data:
                    train_data[image].append(image)
                else:
                    train_data[image] = [image]
    return train_data


def remove_singles(groups):
    """
    Removes groups that contain only a single image.
    """
    return {k: v for k, v in groups.items() if len(v) > 1}


def _post_process_groups_similarity(G, initial_grouped, max_size):
    """
    Splits groups larger than max_size by prioritizing strong connections
    within the community. Uses a Maximum Spanning Tree (MaxST) heuristic
    to identify and break the weakest links first.
    """
    final_groups = {}
    current_truck_id = 1

    for group_name, members in initial_grouped.items():
        if len(members) <= max_size:
            # Group is already small enough, just rename and keep
            final_groups[f"Truck_{current_truck_id}"] = members
            current_truck_id += 1
        else:
            # --- Oversized Group Splitting Logic ---

            # 1. Create a subgraph for the oversized community
            subgraph = G.subgraph(members).copy()

            # 2. Get the edges sorted by weight (strongest first)
            # MaxST edges are those that maximize the total weight.
            # We can approximate the MaxST by just keeping the strongest edges.

            # Get the Maximum Spanning Tree (MaxST) - by negating weights and finding MST
            # Or simpler: get all edges and sort them by weight descending.
            edges = sorted(
                subgraph.edges(data=True), key=lambda x: x[2]["weight"], reverse=True
            )

            # Create a new graph containing only the MaxST-like structure
            # to make component splitting cleaner.
            split_G = nx.Graph()
            split_G.add_nodes_from(subgraph.nodes)

            # Add edges in order of strength until all nodes are connected
            temp_nodes = set()
            for u, v, data in edges:
                if u not in temp_nodes or v not in temp_nodes:
                    split_G.add_edge(u, v, weight=data["weight"])
                    temp_nodes.add(u)
                    temp_nodes.add(v)

            # Now, iterate over the *weakest* edges and remove them until
            # all components are <= max_size.
            # Get all edges of the MST/strongest structure, sorted by weight ascending (weakest first)
            weakest_edges = sorted(
                split_G.edges(data=True), key=lambda x: x[2]["weight"], reverse=False
            )

            # Remove the weakest edges one by one
            for u, v, data in weakest_edges:
                split_G.remove_edge(u, v)

                # Check if the largest resulting component meets the size constraint
                largest_component_size = max(
                    len(c) for c in nx.connected_components(split_G)
                )

                if largest_component_size <= max_size:
                    break  # Stop removing edges

            # 3. Extract the final components (the new groups)
            for component in nx.connected_components(split_G):
                final_groups[f"Truck_{current_truck_id}"] = list(component)
                current_truck_id += 1

    return final_groups


def solve_truck_grouping_max_size_similarity(data, threshold=0.8, max_size=3):
    G = nx.Graph()

    # 1. Build the Graph
    for main_img, matches in data.items():
        print(f"Processing main image: {main_img} with {len(matches)} matches.")
        G.add_node(main_img)
        for match_img, score_str in matches:
            try:
                score = float(score_str)
            except:
                continue

            # Add an edge if the similarity score meets the threshold
            if score >= threshold:
                # Ensure the edge is added in both directions (if not symmetric in data)
                G.add_edge(main_img, match_img, weight=score)

    # 2. Community Detection (Louvain)
    # This step groups based on optimal modularity
    partition = community_louvain.best_partition(G, weight="weight")

    # 3. Convert to Initial Format
    initial_grouped = {}
    for node, group_id in partition.items():
        # Temporarily use the partition ID for grouping
        initial_grouped.setdefault(f"Temp_Group_{group_id}", []).append(node)

    # 4. Post-Process to Enforce Max Size Constraint based on Similarity
    final_grouped = _post_process_groups_similarity(G, initial_grouped, max_size)

    return final_grouped


def evaluate_grouping(pred_groups, truth_groups):
    # 1. Create mappings from image -> group_id
    def get_map(groups):
        mapping = {}
        for gid, images in groups.items():
            for img in images:
                mapping[img] = gid
        return mapping

    pred_map = get_map(pred_groups)
    truth_map = get_map(truth_groups)

    # 2. Identify all unique images
    all_images = sorted(list(set(pred_map.keys()) | set(truth_map.keys())))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # 3. Iterate over all unique pairs
    # This compares every pair of images to see if they are correctly grouped or separated
    for img1, img2 in itertools.combinations(all_images, 2):
        # Check Ground Truth
        # Two images are "same" in truth only if both exist in the map and have same ID.
        # Images not in the map are considered singletons (unique IDs).
        is_same_truth = (
            img1 in truth_map
            and img2 in truth_map
            and truth_map[img1] == truth_map[img2]
        )

        # Check Prediction
        is_same_pred = (
            img1 in pred_map and img2 in pred_map and pred_map[img1] == pred_map[img2]
        )

        if is_same_truth and is_same_pred:
            tp += 1
        elif not is_same_truth and is_same_pred:
            fp += 1
        elif is_same_truth and not is_same_pred:
            fn += 1
        else:
            tn += 1

    print("-" * 30)
    print("Pairwise Grouping Evaluation:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 30)


# --- 2. UI Class ---
class TruckGalleryApp:
    def __init__(self, root, groups_data):
        self.root = root
        self.root.title("Truck Identification Gallery")
        self.root.geometry("900x800")

        # Keep references to images so they don't get garbage collected
        self.image_refs = []

        self.groups = groups_data
        # Convert dict to list for indexing
        self.group_list = list(self.groups.items())
        self.current_group_index = 0

        self.setup_ui()
        self.display_current_group()

    def setup_ui(self):
        # --- Control Frame (Top) ---
        self.control_frame = tk.Frame(self.root, bg="#d0d0d0", pady=10)
        self.control_frame.pack(fill=tk.X, side=tk.TOP)

        self.btn_prev = tk.Button(
            self.control_frame,
            text="<< Previous",
            command=self.prev_group,
            font=("Arial", 12),
        )
        self.btn_prev.pack(side=tk.LEFT, padx=20)

        self.lbl_status = tk.Label(
            self.control_frame, text="", font=("Arial", 14, "bold"), bg="#d0d0d0"
        )
        self.lbl_status.pack(side=tk.LEFT, expand=True)

        self.btn_next = tk.Button(
            self.control_frame,
            text="Next >>",
            command=self.next_group,
            font=("Arial", 12),
        )
        self.btn_next.pack(side=tk.RIGHT, padx=20)

        # --- Main Content (Scrollable) ---
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=1)

        self.canvas = tk.Canvas(self.main_frame, bg="#e6e6e6")
        self.scrollbar = ttk.Scrollbar(
            self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )

        self.scrollable_frame = tk.Frame(self.canvas, bg="#e6e6e6")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def prev_group(self):
        if self.current_group_index > 0:
            self.current_group_index -= 1
            self.display_current_group()

    def next_group(self):
        if self.current_group_index < len(self.group_list) - 1:
            self.current_group_index += 1
            self.display_current_group()

    def load_thumbnail(self, path, size=(350, 350)):
        """
        Tries to load image from disk.
        If not found, creates a placeholder with text.
        """
        try:
            # Try to open the actual file
            if os.path.exists(path):
                img = Image.open(path)
                img.thumbnail(size)
                return ImageTk.PhotoImage(img)
            else:
                raise FileNotFoundError
        except Exception:
            # Create a placeholder if file missing
            img = Image.new("RGB", size, color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            # Draw filename in center
            short_name = path.split("/")[-1]
            d.text((10, 60), short_name, fill=(255, 255, 255))
            return ImageTk.PhotoImage(img)

    def display_current_group(self):
        # Clear previous widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.image_refs = []  # Clear old references

        if not self.group_list:
            tk.Label(
                self.scrollable_frame, text="No groups found.", bg="#e6e6e6"
            ).pack()
            return

        truck_id, images = self.group_list[self.current_group_index]

        # Update Status Label
        self.lbl_status.config(
            text=f"Group {self.current_group_index + 1} of {len(self.group_list)}: {truck_id} ({len(images)} images)"
        )

        # Update Button States
        self.btn_prev.config(
            state=tk.NORMAL if self.current_group_index > 0 else tk.DISABLED
        )
        self.btn_next.config(
            state=(
                tk.NORMAL
                if self.current_group_index < len(self.group_list) - 1
                else tk.DISABLED
            )
        )

        # Frame for ONE truck group
        group_frame = tk.LabelFrame(
            self.scrollable_frame,
            text=f" {truck_id} ",
            font=("Helvetica", 12, "bold"),
            bg="white",
            bd=2,
        )
        group_frame.pack(fill=tk.X, padx=20, pady=10, expand=True)

        # Grid for images inside this group
        # We will put 5 images per row
        row_idx = 0
        col_idx = 0
        max_cols = 5

        for img_path in images:
            # 1. Load Image
            img_path = "TruckReidDataset/" + img_path
            photo = self.load_thumbnail(img_path)
            self.image_refs.append(photo)  # Keep reference!

            # 2. Container for Image + Label
            item_frame = tk.Frame(group_frame, bg="white")
            item_frame.grid(row=row_idx, column=col_idx, padx=10, pady=10)

            # 3. The Image
            lbl_img = tk.Label(item_frame, image=photo, bg="white")
            lbl_img.pack()

            # 4. The Filename Text
            clean_name = img_path.split("/")[-1]
            lbl_text = tk.Label(
                item_frame, text=clean_name, font=("Arial", 8), bg="white"
            )
            lbl_text.pack()

            # 5. Grid Logic
            col_idx += 1
            if col_idx >= max_cols:
                col_idx = 0
                row_idx += 1


# --- 3. Data & Run ---
if __name__ == "__main__":
    raw_data_res = read_results("results1.txt")
    group_data = remove_singles(
        solve_truck_grouping_max_size_similarity(
            raw_data_res, threshold=0.76, max_size=3
        )
    )

    group_data_train = remove_singles(
        read_train("TruckReidDataset/train.txt")
    )  # Just to ensure file exists

    group_data_train.pop("reid", None)  # Remove header if present

    evaluate_grouping(group_data, group_data_train)

    # Ensure this script is in the folder ABOVE 'train/', or update paths in raw_data
    root = tk.Tk()
    app = TruckGalleryApp(root, group_data)
    root.mainloop()
