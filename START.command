#!/bin/bash
# Avvia un server locale e apre la demo nel browser
cd "$(dirname "$0")"
PORT=8741
( sleep 1 && open "http://localhost:$PORT" ) &
echo "Server attivo su http://localhost:$PORT — chiudi questa finestra (Ctrl+C) per fermarlo."
echo "Transizione : http://localhost:$PORT/index.html"
echo "Try-on      : http://localhost:$PORT/tryon.html"
python3 -m http.server $PORT
