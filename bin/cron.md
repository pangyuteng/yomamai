### cron

#### run every saturday at 10:01 am.
1 10 * * 6 bash ~/scisoft/numerai/yomamai/bin/run.sh >> ~/scisoft/numerai/yomamai/bin/out.txt 2>&1

#### debugging
*/5 * * * * bash ~/scisoft/numerai/yomamai/bin/run.sh >> ~/scisoft/numerai/yomamai/bin/out.txt 2>&1

### check if cron is running on time.
cat /var/log/syslog | grep CRON


### CUDA Toolkit add to ./bashrc or activate
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}
