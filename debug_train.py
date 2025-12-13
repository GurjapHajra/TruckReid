
def read_train(train_path):
    try:
        open(train_path, "r")  # Ensure file exists
    except FileNotFoundError:
        print("File not found")
        return {}
        
    train_data = {}
    count = 0
    skipped = 0
    skipped_reasons = {"len<4": 0, "reid_empty": 0, "exception": 0}
    
    with open(train_path, "r") as f:
        for line in f:
            count += 1
            try:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    skipped += 1
                    skipped_reasons["len<4"] += 1
                    if skipped < 5:
                        print(f"Skipped line {count} (len<4): {line[:50]}...")
                    continue
                reid = parts[1]
                image = parts[3]

                if reid != "":
                    if reid in train_data:
                        train_data[reid].append(image)
                    else:
                        train_data[reid] = [image]
                else:
                    skipped += 1
                    skipped_reasons["reid_empty"] += 1
                    if skipped_reasons["reid_empty"] < 5:
                        print(f"Skipped line {count} (reid_empty): {line[:50]}...")
            except Exception as e:
                skipped += 1
                skipped_reasons["exception"] += 1
                pass
    print(f"Total lines: {count}")
    print(f"Skipped: {skipped}")
    print(f"Skipped reasons: {skipped_reasons}")
    print(f"Total groups: {len(train_data)}")
    total_images = sum(len(v) for v in train_data.values())
    print(f"Total images: {total_images}")
    
    return train_data

if __name__ == "__main__":
    read_train("TruckReidDataset/train.txt")
