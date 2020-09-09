## Building a TFRecords Dataset for Image Segmentation

Assumes the channel-wise mean and standard deviation have been computed over the dataset and stored in a `.json` file. 


### Create a dictionary for each image/segmentation pair

We want a list of dictionaries, one for each image/segmentation pair. Should include all relevant information including file locations, image dimensions, and labels.

Steps
- Extract list of image file names, shuffle list
- For each file in the list of files ...
    - Load the image into an array
    - Load the segmentation mask (same file name but .png instead of .jpg), convert to array and cast as `np.uint8`
    - Get dimensions of image and mask
    - Parse the file name to get the breed, and breed ID
    - Store location of image and mask, as well as the image dimensions and labels, inside of a dictionary
    - Append the dictionary to a list
    
    
### Use dictionary to serialize dataset and store as TFRecord

Iterate over list of dictionaries 
- Load image and mask arrays
- Perform preprocessing (this example normalizes color channels)
- Serialize image and mask into byte-strings
- Write into a file using a `tf.io.TFRecordsWriter`


### Verify by reading from TFRecord

Important note: Need to manualy specify the image depth and mask depth in `read_tfrecord()`, as well as data type. Otherwise model with throw an error during training. 