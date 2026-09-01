sed -i 's/\r$//' "$0"

set -euo pipefail

start_time=$(date +%s)

dataname=(
27d42437168ccd7ddd75f724c0ccbe00
)


mkdir -p log   
logfile="log/$(date +"%Y-%m-%d_%H-%M-%S")_shapenetcars.log"

echo "===== $(date) : 开始全部训练，日志写入 ${logfile} =====" >> "$logfile"

for h in "${dataname[@]}"; do 
    echo "===== $(date) : 启动 shapenetCars ${h} =====" >> "$logfile"
    stdbuf -oL -eL python run.py \
        --gpu 0 \
        --conf "confs/shapenet_cars.conf" \
        --dir "shapenetcars" \
        --dataname "$h" \
        2>&1 | tee -a "$logfile"
    echo "===== $(date) : 结束 shapenetCars ${h} =====" >> "$logfile"
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))


python tools/log2csv.py "$logfile"
python tools/log2xlsx.py "$logfile"
