#!/bin/bash
# Preprocess script for the Messidor-Original data set.

# Assumes that the data set resides in ./data/messidor.

messidor_download_url="http://webeye.ophth.uiowa.edu/abramoff/messidor-2.zip"
messidor_dir="./data/messidor2"
messidor_path="$messidor_dir/messidor-2.zip"
grades_path="./vendor/messidor/abramoff-messidor-2-refstandard-jul16.csv"
default_output_dir="$messidor_dir/bin2"

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
check_parameters "$#" "$strip_params" "output"

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

messidor_size=$(ls -s $messidor_path 2>/dev/null | cut -d " " -f1)

if [ !ls "$messidor_path" >/dev/null 2>&1 ] || \
    [[ $messidor_size -ne 3398280 ]]; then
  echo "Downloading messidor-2.zip. This may take a while..."
  curl -L0 "$messidor_download_url" --output "$messidor_path"
fi

count_files=$(ls $messidor_dir/Messidor-2 2>/dev/null | wc -l)

if [ !ls "$messidor_dir/Messidor-2" >/dev/null 2>&1 ] || \
    [[ $count_files -ne 1757 ]]; then
  # Check if unzip has been installed.
  dpkg -l | grep unzip
  if [ $? -gt 0 ]; then
    echo "Please install unzip: apt-get/yum install unzip" >&2
    exit 1
  fi

  if [[ $count_files -ne 1757 ]]; then
    echo "Messidor-2 wasn't unpacked properly before"
  fi

  echo "Unpacking messidor-2.zip"
  unzip "$messidor_path" -d "$messidor_dir" || exit 1
fi

# Copying labels file from vendor to data directory.
cp "$grades_path" "$messidor_dir/labels.csv"

# Preprocess the data set and categorize the images by labels into
#  subdirectories.
python preprocess_messidor2.py --data_dir="$messidor_dir" || exit 1

echo "Preparing data set..."
mkdir -p "$output_dir/0" "$output_dir/1"

echo "Moving images to new directories..."
find "$messidor_dir/0" -iname "*.jpg" -exec mv {} "$output_dir/0/." \;
find "$messidor_dir/1" -iname "*.jpg" -exec mv {} "$output_dir/1/." \;

# Convert the data set to tfrecords.
echo "Converting data set to tfrecords..."
git submodule update --init

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$output_dir" \
       --num_shards=2 || \
    { echo "Submodule not initialized. Run git submodule update --init";
      exit 1; }

echo "Cleaning up..."
rm -r "$messidor_path" "$messidor_dir/Messidor-2" "$messidor_dir/labels.csv"

echo "Done!"
exit

# References:
# [1] http://www.adcis.net/en/Download-Third-Party/Messidor.html
