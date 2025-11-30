import networkx as nx
from networkx.algorithms import community
import community.community_louvain as community_louvain


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
            reid = parts[1]
            image = parts[3]

            if reid != "":
                if reid in train_data:
                    train_data[reid].append(image)
                else:
                    train_data[reid] = [image]
    return train_data


def solve_truck_grouping(data, threshold=0.8):
    G = nx.Graph()

    for main_img, matches in data.items():
        G.add_node(main_img)
        for match_img, score_str in matches:
            try:
                score = float(score_str)
            except:
                continue
            if score >= threshold:
                G.add_edge(main_img, match_img, weight=score)

    # --- MUCH BETTER COMMUNITY DETECTOR ---
    partition = community_louvain.best_partition(G, weight="weight")

    # --- Convert to your desired format ---
    grouped = {}
    for node, group_id in partition.items():
        grouped.setdefault(f"Truck_{group_id+1}", []).append(node)

    return grouped


# print(read_train("TruckReidDataset\\train.txt").values())
# print(read_results("results.txt"))
# results = read_results("results.txt")

# print(solve_truck_grouping(results, threshold=0.76))


import tkinter as tk
from tkinter import ttk, font
import networkx as nx
from networkx.algorithms import community
from PIL import Image, ImageTk, ImageDraw  # Requires: pip install pillow
import os


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
    # REPLACE THIS WITH YOUR REAL DATA
    raw_data_res = read_results("results1.txt")
    group_data = solve_truck_grouping(raw_data_res, threshold=0.76)

    group_data_train = read_train(
        "TruckReidDataset/train.txt"
    )  # Just to ensure file exists

    del group_data_train["reid"]  # Remove header if present

    # Ensure this script is in the folder ABOVE 'train/', or update paths in raw_data
    root = tk.Tk()
    app = TruckGalleryApp(root, group_data)
    root.mainloop()
