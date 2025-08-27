import os 

def yolo_writer(annots, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for annot in annots:
            f.write(" ".join(str(x) for x in annot) + "\n")
    