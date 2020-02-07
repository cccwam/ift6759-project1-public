#!/bin/bash
# Example usage:
#   ./salloc_auto.sh 2:00:00
# Definition:
#   ./salloc_auto.sh <(optional) time to reserve h:mm:ss>
# Summary:
#   A helper function for launching an interactive
#   session on a k80 gpu node using salloc.
#   Uses the class's reservation if it is available.

time="2:00:00"
if [ -z "$1" ]
then
  echo "Time unspecified. Defaulting to reservation time of $time"
else
  time="$1"
fi

specs="--gres=gpu:k80:1 --time=$time"

echo "Opening an interactive session."
{
  # Try
  echo "Using the class's reservation..."
  salloc $specs --reservation=IFT6759_$(date +%Y-%m-%d)
} || {
  # Catch
  echo "Skipping the reservation..."
  salloc $specs
}
