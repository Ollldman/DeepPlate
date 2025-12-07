import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
import numpy as np

def plot_image(
    original: Union[np.ndarray, np.generic, list, tuple],
    transformed: Union[np.ndarray, np.generic, list, tuple],
    figsize: Tuple[float, float] = (5, 2.5)
) -> None:
    """
    Display a side-by-side comparison of an original and a transformed image.

    This utility function is intended for debugging and visual inspection of image
    transformations (e.g., data augmentation). It plots two images horizontally:
    the original input and its transformed version.

    Parameters
    ----------
    original : array-like
        The original image array. Must be compatible with matplotlib.pyplot.imshow,
        such as a NumPy array of shape (H, W) or (H, W, C).
    transformed : array-like
        The transformed image array, typically the result of applying augmentation
        or preprocessing to `original`. Must also be imshow-compatible.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default is (5, 2.5).

    Returns
    -------
    None
        This function displays the plot but does not return any value.

    Notes
    -----
    - Both images are displayed without axes and with grayscale colormap (`cmap='gray'`).
      If your image is RGB or has more than one channel and you want to preserve color,
      consider modifying or removing the `cmap='gray'` argument.
    - The function uses `plt.show()`, so it is best suited for interactive environments
      like Jupyter notebooks or scripts with GUI backends enabled.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Original image
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')

    # Transformed image
    ax2.imshow(transformed, cmap='gray')
    ax2.set_title('Transformed')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()