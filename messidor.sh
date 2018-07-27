#!/bin/bash
# Preprocess script for the Messidor-Original data set.

# Assumes that the data set resides in ./data/messidor.

messidor_dir="./data/messidor"
default_output_dir="$messidor_dir/bin2"
grad_grades="./vendor/messidor/messidor_gradability_grades.csv"

check_parameters()
{
  if [ "$1" -ge 3 ]; then
    echo "Illegal number of parameters".
    exit 1
  fi
  if [ "$1" -ge 1 ]; then
    for param in $2; do
      if [ $(echo "$3" | grep -c -- "$param") -eq 0 ]; then
        echo "Unknown parameter $param."
        exit 1
      fi
    done
  fi
  return 0
}

strip_params=$(echo "$@" | sed "s/--\([a-z]\+\)\(=\(.\+\)\)\?/\1/g")
check_parameters "$#" "$strip_params" "output only_gradable"

# Get output directory from parameters.
output_dir=$(echo "$@" | sed "s/.*--output=\([^ ]\+\).*/\1/g")

# Check if output directory is valid.
if ! [[ "$output_dir" =~ ^[^-]+$ ]]; then
  output_dir=$default_output_dir
fi

if ls "$output_dir" >/dev/null 2>&1; then
  echo "Dataset is already located in $output_dir."
  echo "Specify another output directory with the --output flag."
  exit 1
fi

# Confirm the Basexx .zip files and annotations .xls files are present.
xls_count=$(find "$messidor_dir" -maxdepth 1 -iname "Annotation_Base*.xls" | wc -l)
zip_count=$(find "$messidor_dir" -maxdepth 1 -iname "Base*.zip" | wc -l)

if [ $xls_count -ne 12 ]; then
  echo "$messidor_dir does not contain any all annotation files!"
  exit 1
fi

if [ $zip_count -ne 12 ]; then
  echo "$messidor_dir does not contain all Basexx zip files!"
  exit 1
fi

# Preprocess the data set and categorize the images by labels into
#  subdirectories.
python preprocess_messidor.py --data_dir="$messidor_dir" || exit 1

# Remove ungradable images if needed.
if echo "$@" | grep -F -c -- "--only_gradable" >/dev/null; then
  echo "Remove ungradable images"
  cat "$grad_grades" | while read tbl; do
    if [[ "$tbl" =~ ^.*0$ ]]; then
      file=$(echo "$tbl" | sed "s/\(.*\) 0/\1/")
      find "$messidor_dir"/[0-3] -iname "$file*" -delete
    fi
  done
fi

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
20051202_55498_0400_PP.jpg" | tr " " "\n" | xargs -I% find "$messidor_dir" -name % -delete

# 31 August 2016: Erratum in Base11 Excel file
find "$messidor_dir/3" -name "20051020_63045_0100_PP.jpg" -exec mv {} "$messidor_dir/0/." \;

# 24 October 2016: Erratum in Base11 and Base 13 Excel files
find "$messidor_dir/1" -name "20051020_64007_0100_PP.jpg" -exec mv {} "$messidor_dir/3/." \;
find "$messidor_dir/3" -name "20051020_63936_0100_PP.jpg" -exec mv {} "$messidor_dir/1/." \;
find "$messidor_dir/2" -name "20060523_48477_0100_PP.jpg" -exec mv {} "$messidor_dir/3/." \;

echo "Preparing data set..."
mkdir -p "$output_dir/0" "$output_dir/1"

echo "Moving images to new directories..."
find "$messidor_dir/"[0-1] -iname "*.jpg" -exec mv {} "$output_dir/0/." \;
find "$messidor_dir/"[2-3] -iname "*.jpg" -exec mv {} "$output_dir/1/." \;

echo "Removing old directories..."
rmdir "$messidor_dir/"[0-3]

# Convert the data set to tfrecords.
echo "Converting data set to tfrecords..."
git submodule update --init

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$output_dir" \
       --num_shards=2 || \
    { echo "Submodule not initialized. Run git submodule update --init";
      exit 1; }

echo "Done!"
exit

# References:
# [1] http://www.adcis.net/en/Download-Third-Party/Messidor.html
