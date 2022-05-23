# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조
# sh run.sh => 실행
#!/bin/bash

echo "Starting server"
python server.py &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait