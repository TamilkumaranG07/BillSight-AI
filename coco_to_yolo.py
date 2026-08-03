import json
import shutil
from pathlib import Path


# ==========================================================
# CHANGE THESE PATHS
# ==========================================================

SOURCE_ROOT = Path(r"C:\Project Phase 1\dataset")

OUTPUT_ROOT = Path(r"C:\Project Phase 1\yolo_dataset")


# Source folder -> YOLO folder
SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test"
}


# ==========================================================
# CONVERT ONE DATASET SPLIT
# ==========================================================

def convert_split(source_split, output_split):

    source_folder = SOURCE_ROOT / source_split

    json_path = (
        source_folder
        / "_annotations.coco.json"
    )

    output_image_folder = (
        OUTPUT_ROOT
        / "images"
        / output_split
    )

    output_label_folder = (
        OUTPUT_ROOT
        / "labels"
        / output_split
    )

    # Create output folders
    output_image_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_label_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 60)

    print(
        f"Converting "
        f"{source_split} "
        f"to "
        f"{output_split}"
    )

    print("=" * 60)

    # Check JSON file
    if not json_path.exists():

        print(
            f"ERROR: JSON not found:\n"
            f"{json_path}"
        )

        return None

    # Load COCO annotations
    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        coco = json.load(file)

    images = coco.get(
        "images",
        []
    )

    annotations = coco.get(
        "annotations",
        []
    )

    categories = coco.get(
        "categories",
        []
    )

    print(
        "Images:",
        len(images)
    )

    print(
        "Annotations:",
        len(annotations)
    )

    print(
        "Classes:",
        len(categories)
    )

    # ======================================================
    # CREATE COCO CLASS ID -> YOLO CLASS ID
    # ======================================================

    categories = sorted(
        categories,
        key=lambda category:
        category["id"]
    )

    category_map = {}

    for yolo_id, category in enumerate(
        categories
    ):

        coco_id = category["id"]

        category_map[
            coco_id
        ] = yolo_id

        print(
            f"COCO ID {coco_id} "
            f"-> YOLO ID {yolo_id} "
            f"({category['name']})"
        )

    # ======================================================
    # GROUP ANNOTATIONS BY IMAGE
    # ======================================================

    annotations_by_image = {}

    for annotation in annotations:

        # Ignore crowd annotations
        if annotation.get(
            "iscrowd",
            0
        ) == 1:

            continue

        image_id = (
            annotation[
                "image_id"
            ]
        )

        if image_id not in (
            annotations_by_image
        ):

            annotations_by_image[
                image_id
            ] = []

        annotations_by_image[
            image_id
        ].append(
            annotation
        )

    # ======================================================
    # CONVERT IMAGES AND LABELS
    # ======================================================

    converted_images = 0

    converted_boxes = 0

    missing_images = 0

    for image_info in images:

        image_id = (
            image_info["id"]
        )

        image_name = (
            image_info["file_name"]
        )

        image_width = (
            image_info["width"]
        )

        image_height = (
            image_info["height"]
        )

        # Images are directly in:
        # train/
        # valid/
        # test/

        source_image = (
            source_folder
            / image_name
        )

        if not source_image.exists():

            print(
                f"WARNING: "
                f"Image not found: "
                f"{source_image}"
            )

            missing_images += 1

            continue

        # Copy image
        destination_image = (
            output_image_folder
            / source_image.name
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        # Create matching YOLO label
        label_path = (
            output_label_folder
            / (
                source_image.stem
                + ".txt"
            )
        )

        yolo_labels = []

        image_annotations = (
            annotations_by_image.get(
                image_id,
                []
            )
        )

        for annotation in (
            image_annotations
        ):

            category_id = (
                annotation[
                    "category_id"
                ]
            )

            if category_id not in (
                category_map
            ):

                continue

            # COCO:
            # x_top_left,
            # y_top_left,
            # width,
            # height

            x, y, width, height = (
                annotation[
                    "bbox"
                ]
            )

            # Ignore invalid boxes
            if (
                width <= 0
                or height <= 0
            ):

                continue

            # Convert to center coordinates
            x_center = (
                x
                + width / 2
            )

            y_center = (
                y
                + height / 2
            )

            # Normalize to 0-1
            x_center /= (
                image_width
            )

            y_center /= (
                image_height
            )

            width /= (
                image_width
            )

            height /= (
                image_height
            )

            # Clip values
            x_center = max(
                0,
                min(
                    1,
                    x_center
                )
            )

            y_center = max(
                0,
                min(
                    1,
                    y_center
                )
            )

            width = max(
                0,
                min(
                    1,
                    width
                )
            )

            height = max(
                0,
                min(
                    1,
                    height
                )
            )

            yolo_class_id = (
                category_map[
                    category_id
                ]
            )

            # YOLO format:
            # class x_center y_center width height

            yolo_labels.append(

                f"{yolo_class_id} "

                f"{x_center:.6f} "

                f"{y_center:.6f} "

                f"{width:.6f} "

                f"{height:.6f}"

            )

            converted_boxes += 1

        # Save label file
        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "\n".join(
                    yolo_labels
                )
            )

        converted_images += 1

    print("\nCompleted!")

    print(
        "Images converted:",
        converted_images
    )

    print(
        "Bounding boxes:",
        converted_boxes
    )

    print(
        "Missing images:",
        missing_images
    )

    return categories


# ==========================================================
# CREATE data.yaml
# ==========================================================

def create_data_yaml(categories):

    categories = sorted(
        categories,
        key=lambda category:
        category["id"]
    )

    dataset_path = (
        OUTPUT_ROOT
        .resolve()
        .as_posix()
    )

    yaml_path = (
        OUTPUT_ROOT
        / "data.yaml"
    )

    yaml_text = (
        f"path: {dataset_path}\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(categories)}\n\n"
        "names:\n"
    )

    for yolo_id, category in enumerate(
        categories
    ):

        yaml_text += (
            f"  {yolo_id}: "
            f"{category['name']}\n"
        )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            yaml_text
        )

    print("\n" + "=" * 60)

    print(
        "data.yaml created:"
    )

    print(
        yaml_path.resolve()
    )

    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "\nCOCO TO YOLOv8 "
        "CONVERTER"
    )

    print(
        "Source:",
        SOURCE_ROOT
    )

    print(
        "Output:",
        OUTPUT_ROOT
    )

    categories = None

    for source_split, output_split in (
        SPLITS.items()
    ):

        result = convert_split(
            source_split,
            output_split
        )

        if result is not None:

            categories = result

    if categories:

        create_data_yaml(
            categories
        )

    print(
        "\nConversion completed!"
    )


if __name__ == "__main__":

    main()