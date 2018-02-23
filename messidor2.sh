#!/bin/bash
# Preprocess script for the Messidor2 data set.

# Assumes that the data set resides in ./data/messidor2.

messidor2_dir="./data/messidor2"

# Confirm the Basexx .zip files and annotations .xls files are present.
xls_count=$(find "$messidor2_dir" -maxdepth 1 -iname "Annotation_Base*.xls" | wc -l)
zip_count=$(find "$messidor2_dir" -maxdepth 1 -iname "Base*.zip" | wc -l)

if [ $xls_count -ne 12 ]; then
  echo "$messidor2_dir does not contain any all annotation files!"
  exit 1
fi

if [ $zip_count -ne 12 ]; then
  echo "$messidor2_dir does not contain all Basexx zip files!"
  exit 1
fi

# Preprocess the data set and categorize the images by labels into
#  subdirectories.
python preprocess_messidor2.py --data_dir="$messidor2_dir" || exit 1

# According to [1], we have to correct some duplicate images and
#  grades in the data set.

echo "Correcting data set..."
# 16 August 2017: Image duplicates in Base33
echo "20051202_54744_0400_PP.jpg 20051202_40508_0400_PP.jpg
20051202_41238_0400_PP.jpg 20051202_41260_0400_PP.jpg
20051202_54530_0400_PP.jpg 20051205_33025_0400_PP.jpg
20051202_55607_0400_PP.jpg 20051202_41034_0400_PP.jpg
20051205_35099_0400_PP.jpg 20051202_54555_0400_PP.jpg
20051205_35110_0400_PP.jpg 20051202_54611_0400_PP.jpg
20051202_55498_0400_PP.jpg " | xargs -d" " -I% find "$messidor2_dir" -iname % -delete

# 31 August 2016: Erratum in Base11 Excel file
find "$messidor2_dir/3" -name "20051020_63045_0100_PP.jpg" -exec mv {} "$messidor2_dir/0/." \;

# 24 October 2016: Erratum in Base11 and Base 13 Excel files
find "$messidor2_dir/1" -name "20051020_64007_0100_PP.jpg" -exec mv {} "$messidor2_dir/3/." \;
find "$messidor2_dir/3" -name "20051020_63936_0100_PP.jpg" -exec mv {} "$messidor2_dir/1/." \;
find "$messidor2_dir/2" -name "20060523_48477_0100_PP.jpg" -exec mv {} "$messidor2_dir/3/." \;

echo "Preparing data set..."
mkdir -p "$messidor2_dir/bin2/0" "$messidor2_dir/bin2/1"

echo "Copying to new directories..."
find "$messidor2_dir/"[0-1] -iname "*.jpg" -exec mv {} "$messidor2_dir/bin2/0/." \;
find "$messidor2_dir/"[2-3] -iname "*.jpg" -exec mv {} "$messidor2_dir/bin2/1/." \;

# Convert the data set to tfrecords.
echo "Converting data set to tfrecords..."
git submodule update --init

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$messidor2_dir/bin2" \
       --tfrecord_filename=messidor2 --num_shards=2 || \
    { echo "Submodule not initialized. Run git submodule update --init";
      exit 1; }

echo "Done!"
exit

# References:
# [1] http://www.adcis.net/en/Download-Third-Party/Messidor.html
