# Fibre Diameter Measurement in Pixel

Fibre diameter measurement program based on `opencv`, `scipy` and `torch`. Raw image will be processed and binarized via opencv and a pointNet style model, the skeleton of fibre will then be extracted. The diameter measurement is done based on the skeleton, via `distance_transform_edt()` in `scipy`. 
## Usage
Move all files under the root path of your project, import `fibreMeausre` module via:

    import fibreMeasure

Activate module via:
    
    fibreMeasure.setup()

The default public API for fibreMeasure is `measure()`, it returns the average fibre diameter in pixel for the input image and the processed binary image.

    measure(imgPath: str) -> float, np.ndarray

## License

This project is released under Apache 2.0 license

## Requirements
<ul>
    <li> opencv-python
    <li> numpy
    <li> skimage
    <li> scipy
    <li> torch
    <li> pandas
    <li> sklearn
    <li> tqdm
    <li> matplotlib
</ul>

## Other Notice

Currently no error handling mechanism developed, so be careful to input the correct `imgPath` into the `measure()` function.



