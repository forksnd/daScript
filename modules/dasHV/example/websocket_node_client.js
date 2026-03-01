// npm install ws

const WebSocket = require('ws');
const https = require('https');

// Replace with your actual WebSocket URL
const url = 'wss://127.0.0.1:9999';

// Create an agent that ignores unauthorized SSL certs
const agent = new https.Agent({
  rejectUnauthorized: false
});

const ws = new WebSocket(url, {
  agent: agent
});

ws.on('open', function open() {
  console.log('Connection opened');
  ws.send('hello');
});

ws.on('message', function message(data) {
  console.log('Received:', data.toString());
});

ws.on('close', function close() {
  console.log('Connection closed');
});

ws.on('error', function error(err) {
  console.error('WebSocket error:', err.message);
});
