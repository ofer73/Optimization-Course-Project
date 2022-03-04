squeue -u maorkehati -o "%.65j %.18i %.2t %.10M" | grep optim
echo ""
echo ""
squeue -u maorkehati -o "%.10j %.18i %.2t %.10M" | grep optim | wc -l | awk '{print $1-0}'
