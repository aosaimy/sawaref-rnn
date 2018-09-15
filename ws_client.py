import websocket
from websocket import create_connection
import _thread as thread
import time
import sys
import json

class WebSocketClient:
    vector_size = None
    cache = {}
    def on_message(self, ws, message):
        print(message)

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws):
        print("### closed ###")

    def on_open(self, ws):
        # def run(*args):
        #     while True:
        #         time.sleep(5)
        #         ws.send('{"word":"خالد"}')
        #     time.sleep(1)
        #     ws.close()
        # thread.start_new_thread(run, ())
        print("connected")

    def set_vector_size(self):
        w = self.send("ا")
        self.vector_size = len(w)

    def send(self, word):
        # d = dict({"word":word})
        if word in self.cache:
            return self.cache[word]
        self.ws.send(json.dumps({"word": word,
                               # "embeddings": s,
                               "id": "test"
                               }))
        r = json.loads(self.ws.recv())
        if r["word"] != word:
            print("Error: embedding is not in sync")
            exit()
        self.cache[word] = r["embeddings"]
        return self.cache[word]


    def __getitem__(self, word):
        return self.send(word)


    def __init__(self, path):
        websocket.enableTrace(True)
        # self.ws = websocket.WebSocketApp(path,
        #                           on_message = self.on_message,
        #                           on_error = self.on_error,
        #                           on_close = self.on_close)
        # self.ws.on_open = self.on_open
        self.ws = create_connection(path)
        self.set_vector_size()
        self.ws.on_open = self.on_open

if __name__ == "__main__":
    w = WebSocketClient("ws://localhost:8765/")
    print(w.send("تجربة"))
    w.ws.close()