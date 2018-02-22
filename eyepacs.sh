#!/bin/bash
# Preprocess script for the EyePACS data set from Kaggle.

# Assumes that the data set resides in ./data/eyepacs.

eyepacs_dir="./data/eyepacs"
pool_dir="$eyepacs_dir/pool"

# Confirm the Basexx .zip files and annotations .xls files are present.
train_zip_count=$(find "$eyepacs_dir" -maxdepth 1 -iname "train.zip.00*" | wc -l)
test_zip_count=$(find "$eyepacs_dir" -maxdepth 1 -iname "test.zip.00*" | wc -l)
train_csv_zip=$(find "$eyepacs_dir" -maxdepth 1 -iname "trainLabels.csv.zip" | wc -l)

if [ $train_zip_count -ne 5 ]; then
  echo "$eyepacs_dir does not contain all train.zip files!"
  exit 1
fi

if [ $test_zip_count -ne 7 ]; then
  echo "$eyepacs_dir does not contain all test.zip files!"
  exit 1
fi

if [ $train_csv_zip -ne 1 ]; then
  echo "$eyepacs_dir does not contain trainLabels.csv.zip file!"
  exit 1
fi

echo "Unzip the data set (0/2)..."

# Check if p7zip is installed.
dpkg -l | grep p7zip-full
if [ $? -gt 0 ]; then
  echo "Please install p7zip-full: apt-get/yum install p7zip-full" >&2
  exit 1
fi

# Unzip training set.
#7z e "$eyepacs_dir/train.zip.001" -o"$pool_dir" || exit 1

echo "Unzip the data set (1/2)..."

# Unzip test set.
#7z e "$eyepacs_dir/test.zip.001" -o"$pool_dir" || exit 1

# Copy test labels from vendor to data set folder.
#cp vendor/eyepacs/testLabels.csv.zip "$eyepacs_dir/."

# Unzip labels.
#7z e "$eyepacs_dir/trainLabels.csv.zip" -o"$pool_dir" || exit 1
#7z e "$eyepacs_dir/testLabels.csv.zip" -o"$pool_dir" || exit 1

python preprocess_eyepacs.py --data_dir="$pool_dir"

exit

# Convert the data set to tfrecords.
echo "Converting data set to tfrecords..."
git submodule update --init

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$messidor2_dir" \
       --tfrecord_filename=eyepacs --num_shards=2 || \
    { echo "Submodule not initialized. Run git submodule update --init"; 
      exit 1; }

echo "Done!"
exit

# References:
# [1] http://www.adcis.net/en/Download-Third-Party/Messidor.html
