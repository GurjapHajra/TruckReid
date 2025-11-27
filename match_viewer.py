import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os

# --- CONFIGURATION ---
FILE_PATH = "results.txt"  # Your text file name
IMAGE_DIR = "TruckReidDataset"  # Directory where the 'train' folder is located


class MatchViewerApp:
    def __init__(self, root, data):
        self.root = root
        self.data = data
        self.index = 0

        self.root.title("Image Match Viewer")
        self.root.geometry("800x500")

        # --- Layout Frames ---
        # Top: Titles and Score
        self.info_frame = Frame(root)
        self.info_frame.pack(pady=10)

        self.lbl_score = Label(
            self.info_frame, text="", font=("Arial", 16, "bold"), fg="blue"
        )
        self.lbl_score.pack()

        self.lbl_counter = Label(self.info_frame, text="", font=("Arial", 10))
        self.lbl_counter.pack()

        # Middle: Images (Left = Reference, Right = Match)
        self.image_frame = Frame(root)
        self.image_frame.pack(expand=True, fill="both", padx=20)

        # Reference Image (Left)
        self.ref_container = Frame(self.image_frame)
        self.ref_container.pack(side="left", expand=True)
        Label(
            self.ref_container, text="REFERENCE (1.0)", font=("Arial", 12, "bold")
        ).pack(pady=5)
        self.lbl_ref_img = Label(self.ref_container)
        self.lbl_ref_img.pack()
        self.lbl_ref_name = Label(self.ref_container, text="")
        self.lbl_ref_name.pack()

        # Match Image (Right)
        self.match_container = Frame(self.image_frame)
        self.match_container.pack(side="right", expand=True)
        Label(
            self.match_container, text="MATCH CANDIDATE", font=("Arial", 12, "bold")
        ).pack(pady=5)
        self.lbl_match_img = Label(self.match_container)
        self.lbl_match_img.pack()
        self.lbl_match_name = Label(self.match_container, text="")
        self.lbl_match_name.pack()

        # Bottom: Navigation Buttons
        self.btn_frame = Frame(root)
        self.btn_frame.pack(side="bottom", fill="x", pady=20)

        self.btn_prev = Button(
            self.btn_frame,
            text="<< Previous",
            command=self.prev_match,
            height=2,
            width=15,
        )
        self.btn_prev.pack(side="left", padx=50)

        self.btn_next = Button(
            self.btn_frame, text="Next >>", command=self.next_match, height=2, width=15
        )
        self.btn_next.pack(side="right", padx=50)

        # Bind keyboard arrow keys
        root.bind("<Left>", lambda e: self.prev_match())
        root.bind("<Right>", lambda e: self.next_match())

        # Initial Load
        self.load_current_pair()

    def load_image(self, path):
        """Loads an image or creates a placeholder if missing."""
        full_path = os.path.join(IMAGE_DIR, path)
        try:
            img = Image.open(full_path)
        except (FileNotFoundError, OSError):
            # Create a placeholder image if file not found
            img = Image.new("RGB", (300, 300), color=(200, 200, 200))
            d = ImageDraw.Draw(img)
            d.text((10, 140), "Image Not Found\n" + path, fill=(0, 0, 0))

        # Resize for UI consistency (keep aspect ratio logic could be added here)
        img = img.resize((300, 300))
        return ImageTk.PhotoImage(img)

    def load_current_pair(self):
        item = self.data[self.index]

        # Update Text Info
        self.lbl_score.config(text=f"Match Score: {item['score']}")
        self.lbl_counter.config(text=f"Pair {self.index + 1} of {len(self.data)}")
        self.lbl_ref_name.config(text=item["ref_path"])
        self.lbl_match_name.config(
            text=item["match_path"] if item["match_path"] else "No Matches Found"
        )

        # Update Images
        self.ref_photo = self.load_image(item["ref_path"])
        self.lbl_ref_img.config(image=self.ref_photo)

        if item["match_path"]:
            self.match_photo = self.load_image(item["match_path"])
        else:
            # Create a specific placeholder for "No Match"
            img = Image.new("RGB", (300, 300), color=(240, 240, 240))
            d = ImageDraw.Draw(img)
            d.text((80, 140), "No Match Candidates", fill=(100, 100, 100))
            self.match_photo = ImageTk.PhotoImage(img)

        self.lbl_match_img.config(image=self.match_photo)

        # Button States
        self.btn_prev.config(state="normal" if self.index > 0 else "disabled")
        self.btn_next.config(
            state="normal" if self.index < len(self.data) - 1 else "disabled"
        )

    def next_match(self):
        if self.index < len(self.data) - 1:
            self.index += 1
            self.load_current_pair()

    def prev_match(self):
        if self.index > 0:
            self.index -= 1
            self.load_current_pair()


def parse_file(filepath):
    """Parses file into a flat list of (Ref, Match, Score) for easy navigation."""
    pairs = []
    current_ref = None

    if not os.path.exists(filepath):
        print("File not found! Creating dummy data.")
        return [
            {
                "ref_path": "dummy_ref.jpg",
                "match_path": "dummy_match.jpg",
                "score": 0.99,
            }
        ]

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            path, score = parts[0], float(parts[1])

            if score == 1.0:
                current_ref = path
                # Optional: Add an entry just for the Ref if you want to see orphans
                # Currently we only add it if we find matches or if we handle orphans later
                has_matches = False
            else:
                if current_ref:
                    pairs.append(
                        {"ref_path": current_ref, "match_path": path, "score": score}
                    )
                    has_matches = True

    return pairs


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Parse the data
    flat_data = parse_file(FILE_PATH)

    if not flat_data:
        print("No matches found in the file.")
    else:
        root = tk.Tk()
        app = MatchViewerApp(root, flat_data)
        root.mainloop()
