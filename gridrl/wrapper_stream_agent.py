#!/usr/bin/env python3

"""Wrapper to stream game data to a websocket server."""

from typing import Union,Any
import warnings
import sys
import json
import asyncio
sys.dont_write_bytecode=True

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import websockets
    from gymnasium import Env,Wrapper

WEBSOCKET_SERVER_ADDRESS="ws://localhost:8080/broadcast"

class StreamWrapper(Wrapper):
    """Streamer Wrapper class."""
    def __init__(self,env:Env,stream_metadata:dict,stream_url:Union[str,None]=None,upload_interval:int=250)->None:
        """Constructor."""
        super().__init__(env)
        self.ws_address = WEBSOCKET_SERVER_ADDRESS if stream_url is None else str(stream_url)
        self.stream_metadata = dict(stream_metadata)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.loop.run_until_complete(self.establish_wc_connection())
        self.upload_interval = max(2,int(upload_interval))
        self.steam_step_counter = 0
        self.coord_list = []
        self.env_type=""
        if hasattr(env, "game_selector"):
            self.emulator = env
            self.env_type="gridrl"
        else:
            raise AttributeError("Could not find game!")
    def get_streamed_coordinates(self)->list:
        """Get game coordinates."""
        if hasattr(self.emulator,"get_streamed_coordinates"):
            return self.emulator.get_streamed_coordinates()
        return [0, 0, 0]
    def upload_coords(self)->None:
        """Upload the data."""
        if len(self.coord_list)==0:
            return
        self.loop.run_until_complete(self.broadcast_ws_message(json.dumps({"metadata": self.stream_metadata, "coords": self.coord_list, "env_type": self.env_type})))
        self.coord_list = []
        self.steam_step_counter = 0
    def step(self, action:int=-1)->tuple:
        """Step environment."""
        self.coord_list.append(self.get_streamed_coordinates()[:3])
        self.steam_step_counter += 1
        if self.steam_step_counter >= self.upload_interval:
            self.upload_coords()
        return self.env.step(action)
    def reset(self,*,seed:Union[int,None]=None,options:Union[dict[str, Any],None]=None)->tuple:
        """Reset environment."""
        return self.env.reset(seed)
    def run(self,steps:int,stop_done:bool=True)->None:
        """Run environment."""
        if steps<1:
            while True:
                if self.step(self.agent.predict_action())[2] and stop_done:
                    break
        else:
            for _ in range(steps):
                if self.step(self.agent.predict_action())[2] and stop_done:
                    break
    async def establish_wc_connection(self)->None:
        """Handling websocket connections."""
        try:
            self.websocket=await websockets.connect(self.ws_address)
        except websockets.exceptions.WebSocketException:
            self.websocket=None
    async def broadcast_ws_message(self,message:str)->None:
        """Handling websocket message broadcasting."""
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException:
                self.websocket = None
