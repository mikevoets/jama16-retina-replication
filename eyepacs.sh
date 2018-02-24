#!/bin/bash
# Preprocess script for the EyePACS data set from Kaggle.

# Assumes that the data set resides in ./data/eyepacs.

eyepacs_dir="./data/eyepacs"
pool_dir="$eyepacs_dir/pool"

# From [1].
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

print_usage()
{
  echo "Unpacking and preprocessing script for EyePACS."
  echo "Run with --redistribute if you have ran this before,"
  echo "and only want to redistribute the data set."
  exit 1
}

check_parameters()
{
  if [ "$1" -ge 3 ]; then
    echo "Illegal number of parameters".
    print_usage
  fi
  if [ "$1" -ge 1 ]; then
    for param in $2; do
      if [ $(echo "$3" | grep -c -- "$param") -eq 0 ]; then
        echo "Unknown parameter $param."
        print_usage
      fi
    done
  fi
  return 0
}

strip_params=$(echo "$@" | sed "s/--\([a-z]\+\)\(=\([0-9]\+\)\)\?/\1/g")
check_parameters "$#" "$strip_params" "redistribute seed"

# Get seed from parameters.
shuffle_seed=$(echo "$@" | sed "s/.*--seed=\([0-9]\+\).*/\1/g")

# Replace seed with default seed number if no seed number.
if ! [[ "$shuffle_seed" =~ ^-?[0-9]+$ ]]; then
  shuffle_seed=42
fi

# Skip unpacking if --redistribute parameter is defined.
if [ $(echo "$@" | grep -c -- "--redistribute") -eq 0 ]; then

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
  7z e "$eyepacs_dir/train.zip.001" -o"$pool_dir" || exit 1

  echo "Unzip the data set (1/2)..."

  # Unzip test set.
  7z e "$eyepacs_dir/test.zip.001" -o"$pool_dir" || exit 1

  # Copy test labels from vendor to data set folder.
  cp vendor/eyepacs/testLabels.csv.zip "$eyepacs_dir/."

  # Unzip labels.
  7z e "$eyepacs_dir/trainLabels.csv.zip" -o"$pool_dir" || exit 1
  7z e "$eyepacs_dir/testLabels.csv.zip" -o"$pool_dir" || exit 1

  python preprocess_eyepacs.py --data_dir="$pool_dir"

  # Remove images in pool.
  find "$pool_dir" -maxdepth 1 -iname "*.jpeg" -delete
fi

echo "Finding images..."
for i in {0..4}; do
  k=$(find "$pool_dir/$i" -iname "*.jpg" | wc -l)
  echo "Found $k images in class $i."
done

echo "Creating directories for data sets"
rm -rf "$eyepacs_dir/bin2"
mkdir -p "$eyepacs_dir/bin2/train/0" "$eyepacs_dir/bin2/train/1"
mkdir -p "$eyepacs_dir/bin2/test/0" "$eyepacs_dir/bin2/test/1"
mkdir -p "$eyepacs_dir/bin2/validation"

# Define images for binary (2) class 0.
bin2_0_cnt=48784
bin2_0_tr_cnt=40688
bin2_0=$(
find "$pool_dir/"[0-1] -iname "*.jpg" |
shuf --random-source=<(get_seeded_random "$shuffle_seed") |
head -n "$bin2_0_cnt"
)

# Define images for binary (2) class 1.
bin2_1=$(find "$pool_dir/"[2-4] -iname "*.jpg")
bin2_1_cnt=$(echo $bin2_1 | tr " " "\n" | wc -l)
bin2_1_tr_cnt=16458

distribute_images()
{
  echo "$1" |
  tr " " "\n" |
  $2 -n "$3" |
  xargs -I{} cp "{}" "$4"
}

echo "Gathering images for train set (0/2)"
distribute_images "$bin2_0" head "$bin2_0_tr_cnt" "$eyepacs_dir/bin2/train/0/."

echo "Gathering images for train set (1/2)"
distribute_images "$bin2_1" head "$bin2_1_tr_cnt" "$eyepacs_dir/bin2/train/1/."

echo "Gathering images for test set (0/2)"
distribute_images "$bin2_0" tail "$(expr $bin2_0_cnt - $bin2_0_tr_cnt)" "$eyepacs_dir/bin2/test/0/."

echo "Gathering images for test set (1/2)"
distribute_images "$bin2_1" tail "$(expr $bin2_1_cnt - $bin2_1_tr_cnt)" "$eyepacs_dir/bin2/test/1/."

echo "Converting data set to tfrecords..."
git submodule update --init

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$eyepacs_dir/bin2/train" \
       --tfrecord_filename=eyepacs --num_shards=8 --validation_size=0.2 || \
    { echo "Submodule not initialized. Run git submodule update --init";
      exit 1; }

echo "Moving validation tfrecords to separate folder."
find "$eyepacs_dir/bin2/train" -name "*eyepacs_validation*.tfrecord" -exec mv {} "$eyepacs_dir/bin2/validation/." \;

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$eyepacs_dir/bin2/test" \
       --tfrecord_filename=eyepacs --num_shards=4 || exit 1

echo "Done!"
exit

# References:
# [1] https://stackoverflow.com/questions/41962359/shuffling-numbers-in-bash-using-seed
