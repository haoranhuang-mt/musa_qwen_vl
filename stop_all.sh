#!/bin/bash

set -u
    HOSTFILE=./hostfile
set +u
NODES_NUM=$(awk '{$1=$1;print}' $HOSTFILE | wc -l)
echo "NODES_NUM": $NODES_NUM

for ((i=1;i<=$NODES_NUM;i++ )); do
    ip=`sed -n $i,1p $HOSTFILE|cut -f 1 -d" "`
    echo "IP": $ip
    # For Musa Image
    ssh $ip "pkill -f '/opt/conda/envs/py38/bin/python /opt/conda/envs/py38/bin/torchrun'" 
    # For Cuda Image
    # ssh $ip "pkill -f '/opt/conda/bin/python /opt/conda/bin/torchrun'" 

done