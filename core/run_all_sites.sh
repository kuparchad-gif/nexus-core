#!/bin/bash
# run_all_sites.sh

echo "🚀 Starting Multi-Site Deployment..."

# Start Lilith in background
echo "Starting Lilith on port 8000..."
python lilith_full_boot.py &
LILITH_PID=$!

# Start Nexus in background  
echo "Starting Nexus on port 8001..."
python integrated_nexus_system.py &
NEXUS_PID=$!

# Wait for services to start
sleep 5

# Start ngrok tunnels in background
echo "Starting ngrok tunnels..."
ngrok http 8000 > /dev/null &
NGROK1_PID=$!

ngrok http 8001 > /dev/null &
NGROK2_PID=$!

sleep 3

echo ""
echo "🎉 Sites are running!"
echo "Lilith: Check terminal for ngrok URL (port 8000)"
echo "Nexus:  Check terminal for ngrok URL (port 8001)"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo ""; echo "🛑 Stopping services..."; kill $LILITH_PID $NEXUS_PID $NGROK1_PID $NGROK2_PID; exit' INT

wait