import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages

# Load dataset (labelled)
df = pd.read_csv("product_images.csv")

X = df.drop(columns=["label"]).values
y = df["label"].values

# Ordered category mapping
categories = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Infer image size (28x28 for 784 pixels)
side = int(np.sqrt(X.shape[1]))
assert side * side == X.shape[1], "Pixel count is not a perfect square."

def save_category_pages_to_pdf(
    X, y, label, label_name,
    cols=10, rows=10,  # 10x10 = 100 images per page
    output_pdf_path=None,
    show_first_page=True
):
    idx = np.where(y == label)[0]

    # Limit to first 100 images only
    per_page = cols * rows
    idx = idx[:per_page]

    n_items = len(idx)
    n_pages = 1  # Only one page

    if output_pdf_path is None:
        output_pdf_path = f"category_{label}{label_name.replace(' ', '').lower()}.pdf"

    with PdfPages(output_pdf_path) as pdf:
        for page in range(n_pages):
            start = page * per_page
            end = min((page + 1) * per_page, n_items)
            page_idx = idx[start:end]

            fig = plt.figure(figsize=(cols * 1.2, rows * 1.2))
            for i, image_idx in enumerate(page_idx):
                ax = fig.add_subplot(rows, cols, i + 1)
                img = X[image_idx].reshape(side, side)
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.axis("off")

            fig.suptitle(
                f"Category {label}: {label_name} | Items {start+1}-{end} of {n_items}",
                y=0.995
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Optionally display only the first page (so you don't open 9*many windows)
            if show_first_page and page == 0:
                # Recreate first page for display
                fig = plt.figure(figsize=(cols * 1.2, rows * 1.2))
                for i, image_idx in enumerate(page_idx):
                    ax = fig.add_subplot(rows, cols, i + 1)
                    img = X[image_idx].reshape(side, side)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                    ax.axis("off")
                fig.suptitle(
                    f"Category {label}: {label_name} | Items {start+1}-{end} of {n_items}",
                    y=0.995
                )
                fig.tight_layout()
                plt.show()
                plt.close(fig)

    print(f"Saved: {output_pdf_path} ({n_pages} pages, {n_items} images)")

# Generate PDFs in the correct order
for label in sorted(categories.keys()):
    safe_name = categories[label].replace("/", "-")
    save_category_pages_to_pdf(
        X, y,
        label=label,
        label_name=categories[label],
        cols=10, rows=10,            # 100 images per page
        output_pdf_path=f"{label}{safe_name}.pdf",
        show_first_page=True     # change to True if you want to preview first page
    )
