#!/bin/bash
# Preprocess script for the EyePACS data set from Kaggle.

# Assumes that the data set resides in ./data/eyepacs.

eyepacs_dir="./data/eyepacs"
default_pool_dir="$eyepacs_dir/pool"
default_shuffle_seed=42
default_output_dir="$eyepacs_dir/bin2"
grad_grades="./vendor/eyepacs/eyepacs_gradability_grades.csv"

# From [1].
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

print_usage()
{
  echo ""
  echo "Extracting and preprocessing script for Kaggle EyePACS."
  echo ""
  echo "Optional parameters: --redistribute, --pool_dir, --seed, --only_gradable"
  echo "--redistribute	Redistribute the data set from pool (default: false)"
  echo "--pool_dir	Path to pool folder (default: $default_pool_dir)"
  echo "--seed		Seed number for shuffling before distributing the data set (default: $default_shuffle_seed)"
  echo "--only_gradable Skip ungradable images. (default: false)"
  echo "--output_dir 	Path to output directory (default: $default_output_dir)"
  exit 1
}

check_parameters()
{
  if [ "$1" -ge 6 ]; then
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

if echo "$@" | grep -c -- "-h" >/dev/null; then
  print_usage
fi

strip_params=$(echo "$@" | sed "s/--\([a-z_]\+\)\(=\([^ ]\+\)\)\?/\1/g")
check_parameters "$#" "$strip_params" "redistribute seed pool_dir only_gradable output_dir"

# Get seed from parameters.
shuffle_seed=$(echo "$@" | sed "s/.*--seed=\([0-9]\+\).*/\1/")

# Replace seed with default seed number if no seed number.
if ! [[ "$shuffle_seed" =~ ^-?[0-9]+$ ]]; then
  shuffle_seed=$default_shuffle_seed
fi

# Get pool directory from parameters.
pool_dir=$(echo "$@" | sed "s/.*--pool_dir=\([^ ]\+\).*/\1/")

# Check if output directory is valid.
if ! [[ "$pool_dir" =~ ^[^-]+$ ]]; then
  pool_dir=$default_pool_dir
fi

# Get output directory from parameters.
output_dir=$(echo "$@" | sed "s/.*--output_dir=\([^ ]\+\).*/\1/")

if ! [[ "$output_dir" =~ ^[^-]+$ ]]; then
  output_dir=$default_output_dir
fi

if ls "$pool_dir" >/dev/null 2>&1 && ! echo "$@" | grep -c -- "--redistribute" >/dev/null; then
  echo "Path already exists: $pool_dir."
  echo ""
  echo "If you want to redistribute data sets from the pool, run this "
  echo " with the --redistribute flag."
  echo "If you want to extract and preprocess the images to another pool "
  echo " directory, specify --pool_dir with a non-existing directory."
  exit 1
fi

if ls "$output_dir" >/dev/null 2>&1; then
  echo "Path already exists: $output_dir."
  echo ""
  echo "Specify a non-existing --output_dir if you want to redistribute"
  echo " from the existing pool to another directory, along with "
  echo " the --redistribute flag."
  exit 1
fi

# Skip unpacking if --redistribute parameter is defined.
if ! echo "$@" | grep -c -- "--redistribute" >/dev/null; then
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
  
  # Remove ungradable images if needed.
  if echo "$@" | grep -c -- "--only_gradable" >/dev/null; then
    echo "Remove ungradable images."
    cat "$grad_grades" | while read tbl; do
      if [[ "$tbl" =~ ^.*0$ ]]; then
        file=$(echo "$tbl" | sed "s/\(.*\) 0/\1/")
        find "$pool_dir" -iname "$file*" -delete
      fi
    done
  fi
fi

# Distribution numbers for data sets with ungradable images.
if echo "$@" | grep -c -- "--only_gradable" >/dev/null; then
  bin2_0_cnt=39202
  bin2_0_tr_cnt=31106
  bin2_1_tr_cnt=12582
else
  bin2_0_cnt=48784
  bin2_0_tr_cnt=40688
  bin2_1_tr_cnt=16458
fi

echo "Finding images..."
for i in {0..4}; do
  k=$(find "$pool_dir/$i" -iname "*.jpg" | wc -l)
  echo "Found $k images in class $i."
done

# Define distributions for data sets.
bin2_0=$(
find "$pool_dir/"[0-1] -iname "*.jpg" |
shuf --random-source=<(get_seeded_random "$shuffle_seed") |
head -n "$bin2_0_cnt"
)

bin2_1=$(find "$pool_dir/"[2-4] -iname "*.jpg")
bin2_1_cnt=$(echo $bin2_1 | tr " " "\n" | wc -l)

echo "Creating directories for data sets"
mkdir -p "$output_dir/train/0" "$output_dir/train/1"
mkdir -p "$output_dir/test/0" "$output_dir/test/1"
mkdir -p "$output_dir/validation"

distribute_images()
{
  echo "$1" |
  tr " " "\n" |
  $2 -n "$3" |
  xargs -I{} cp "{}" "$4"
}

echo "Gathering images for train set (0/2)"
distribute_images "$bin2_0" head "$bin2_0_tr_cnt" "$output_dir/train/0/."

echo "Gathering images for train set (1/2)"
distribute_images "$bin2_1" head "$bin2_1_tr_cnt" "$output_dir/train/1/."

echo "Gathering images for test set (0/2)"
distribute_images "$bin2_0" tail "$(expr $bin2_0_cnt - $bin2_0_tr_cnt)" "$output_dir/test/0/."

echo "Gathering images for test set (1/2)"
distribute_images "$bin2_1" tail "$(expr $bin2_1_cnt - $bin2_1_tr_cnt)" "$output_dir/test/1/."

echo "Converting data set to tfrecords..."
git submodule update --init

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$output_dir/train" \
       --tfrecord_filename=eyepacs --num_shards=8 --validation_size=0.2 || \
    { echo "Submodule not initialized. Run git submodule update --init";
      exit 1; }

echo "Moving validation tfrecords to separate folder."
find "$output_dir/train" -name "*eyepacs_validation*.tfrecord" -exec mv {} "$output_dir/validation/." \;
find "$output_dir/train" -maxdepth 1 -iname "*.txt" -exec cp {} "$output_dir/validation/." \;

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$output_dir/test" \
       --tfrecord_filename=eyepacs --num_shards=4 || exit 1

echo "Done!"
exit

# References:
# [1] https://stackoverflow.com/questions/41962359/shuffling-numbers-in-bash-using-seed
