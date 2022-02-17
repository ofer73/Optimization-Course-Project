squeue -u maorkehati -o "%.65j %.18i %.2t %.10M"
echo ""
echo ""
squeue -u maorkehati -o "%.35j %.18i %.2t %.10M" | wc -l | awk '{print $1-1}'
