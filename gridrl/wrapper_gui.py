#!/usr/bin/env python3

"""Wrapper for environment gui play."""

from typing import Union,Any
import sys
import threading
import tkinter as tk
import numpy as np
from PIL import Image,ImageTk
sys.dont_write_bytecode=True

class TkStdoutRedirector:
    """Redirect stdout to tkinter textarea."""
    def __init__(self,text_widget:tk.Text)->None:
        """Constructor."""
        self.textbox=text_widget
        self.textbox.config(state=tk.NORMAL)
    def write(self,txt:str)->None:
        """On stdout write."""
        self.textbox.insert("end",txt)
        self.textbox.see("1.0")
    def flush(self)->None:
        """On stdout flush."""
        self.textbox.delete("1.0",tk.END)

class GuiWrapper:
    """Tkinter wrapper for manual play, not suited for gymnasium interactions."""
    def __init__(self,env_class,env_config:Union[dict,None]=None,
        agent_class:Union[Any,None]=None,agent_args:Union[dict,None]=None,
        starting_event:str="",collected_flags:Union[list,None]=None,
        play_agent:bool=True,redirect_stdout:bool=True,
        *args,**kwargs
    )->None:
        """Constructor."""
        if env_config is None:
            env_config={}
        if collected_flags is None:
            collected_flags=[]
        config=dict(env_config)
        config.update({"from_gui":True,#"log_screen":False,
            "screen_downscale":1,"grayscale_screen":False,
            "max_steps":-1,"infinite_game":True,"movement_max_actions":4})
        if len(starting_event)>0:
            config["starting_event"]=starting_event
        if len(collected_flags)>0:
            config["starting_collected_flags"]=collected_flags
        self.lock=threading.Lock()
        self.pending_inputs=[]
        self.screenshot_on_quit=False
        self.temp_save=None
        if env_class.__class__.__name__=="type":
            self.env=env_class(config,agent_class,agent_args,*args,**kwargs)
        else:
            self.env=env_class
        self.env.hsv_image=False
        self.play_agent=bool(play_agent) if agent_class is not None else False
        self.quitting=False
        self.screen_movement_axis=True
        self.click_actions=self.env.action_directions.reshape(2,2).transpose(1,0).flatten().tolist()+[self.env.action_interact_id]
        self.og_screen_size=np.array(self.env.screen_ndarray().shape,dtype=np.int16)
        screen=np.zeros(self.og_screen_size,dtype=np.uint8,order="C")
        self.og_screen_size=self.og_screen_size[:2]
        self.update_screen_shapes(False)
        self.resized_screen_size=self.upscale*self.og_screen_size
        self.window=tk.Tk()
        self.window.resizable(False,False)
        self.window.title(f"GridRL [{self.env.get_game_name(add_custom=True)}]")
        self.window.bind("<Control-q>",self.quit)
        self.canvas=tk.Canvas(self.window,
            height=self.resized_screen_size[0],width=self.resized_screen_size[1])
        self.canvas.grid(column=0,row=0,columns=1,rows=1)
        self.canvas_image=self.canvas.create_image(0,0,anchor="nw")
        self.canvas.bind("<1>",self.screen_clicked)
        self.update_screen_content(screen)

        self.window.bind("<r>",self.reset_state)
        self.window.bind("<l>",self.load_state)
        self.window.bind("<s>",self.save_state)
        self.window.bind("<g>",self.change_gui_render)
        self.window.bind("<p>",self.follow_agent_path)
        self.buttons_lookup={f"{self.env.get_powerup_button_text(i+1,True)}":i for i in range(self.env.all_actions_count)}
        for k,v in self.buttons_lookup.items():
            self.window.bind(k,lambda a=v:self.button_clicked(a))
        self.redirect_stdout=bool(redirect_stdout) and self.env.env_id==0
        self.print_commands()
        self.old_stdout=sys.stdout
        if self.redirect_stdout:
            textbox=tk.Text(self.window,wrap="word",height=8,width=0)
            textbox.grid(column=0,row=1,rowspan=1,columnspan=1,sticky="NSWE",padx=5,pady=5)
            sys.stdout=TkStdoutRedirector(textbox)
    def print_commands(self):
        """Prints input hotkeys."""
        print(f"{self.env.get_commands_text()}\n\tScreen click:\tMove to location and interact\n"
            "\tUI buttons:\t[R] Reset to initial state - [L] Load temp state - [S] Save temp state\n"
            "\t\t\t[G] Change GUI render - [P] Follow agent path\n" f"{'='*32}"
        )
    def update_screen_shapes(self,update_og_size:bool=True)->None:
        """Updates screen shapes constants."""
        if update_og_size:
            self.og_screen_size=np.array(self.env.screen_ndarray().shape[:2],dtype=np.int16)
        self.upscale=self.env.default_monocromatic_screen_scale*self.env.screen_downscale if self.env.use_gfx_image else self.env.tile_size
        if update_og_size:
            self.resized_screen_size=self.upscale*self.env.centered_screen_size
    def quit(self,*args,**kwargs)->None:
        """Prepare GUI termination."""
        self.quitting=True
        if self.redirect_stdout:
            sys.stdout=self.old_stdout
        if self.screenshot_on_quit:
            with self.lock:
                self.env.save_run_gif()
        with self.lock:
            self.env.close()
        self.window.after(250,self.after_quit)
    def after_quit(self)->None:
        """Quit the GUI."""
        self.window.destroy()
    def step(self,force:bool=False)->None:
        """Environment step with input handling."""
        if self.quitting:
            return
        used_inputs=[-1]
        done=False
        with self.lock:
            if len(self.pending_inputs)>0:
                used_inputs=list(self.pending_inputs[::-1])
                self.pending_inputs=[]
        if force or used_inputs[0]!=-1:
            for action in used_inputs:
                if self.env.step(action if action>=0 else self.env.action_nop_id)[2]:
                    done=True
                    break
            self.refresh_screen()
        if done:
            self.quit()
        elif not self.quitting:
            self.window.after(50,self.step)
    def refresh_screen(self)->None:
        """Refresh all UI components."""
        with self.lock:
            self.update_screen_content(self.env.screen_ndarray())
            sys.stdout.flush()
            if self.env.agent is not None and not self.play_agent:
                self.env.agent.reset_scheduled_actions()
                self.env.agent.predict_action()
            if self.redirect_stdout:
                print(self.env.get_debug_text())
    def button_clicked(self,action:Union[tk.Event,str,None]=None,*args,**kwargs)->None:
        """Handle action_space input button clicks."""
        if action is None:
            return
        if isinstance(action,tk.Event):
            val_action=self.buttons_lookup.get(action.char,None)
            if val_action is None:
                if action.state==4:
                    act_key=self.env.get_powerup_button_text(10+int(action.keysym)%10,True)
                    val_action=self.buttons_lookup.get(act_key,None)
                if val_action is None:
                    return
            action=val_action
        with self.lock:
            if len(self.pending_inputs)==0 or (len(self.pending_inputs)<1 and action!=self.pending_inputs[-1]):
                self.pending_inputs.append(action)
    def get_coordinates_from_tk_event(self,event:tk.Event)->np.ndarray:
        """Converts event click position to coordinates."""
        return (np.array([event.y,event.x],dtype=np.int16)/(self.upscale*(self.og_screen_size//self.env.centered_screen_size))-0.5).round(0).clip(0,self.env.centered_screen_size.max()).astype(np.int16)
    def screen_clicked(self,event:Union[tk.Event,None]=None,*args,**kwargs)->None:
        """Converts screen click to action_space inputs."""
        if event is None:
            return
        with self.lock:
            if self.env.game_state["menu_type"] in {0,2}:
                swap_axis=int(self.screen_movement_axis)
                self.screen_movement_axis=not self.screen_movement_axis
                movement_offs=self.get_coordinates_from_tk_event(event)-self.env.player_screen_position
                if movement_offs[0]<self.env.player_screen_bounds[1] and movement_offs[1]<self.env.player_screen_bounds[3]:
                    self.pending_inputs+=[self.click_actions[4]]+[self.click_actions[(i^int(swap_axis))*2+(0 if s>0 else 1)] for i,s in enumerate(movement_offs[slice(None,None,-1 if swap_axis else None)]) for j in range(np.abs(s))]
    def reset_state(self,*args,**kwargs)->None:
        """Reset the game to the initial state."""
        with self.lock:
            self.env.reset()
        self.refresh_screen()
    def load_state(self,*args,**kwargs)->None:
        """Load the temp state."""
        if self.temp_save is None:
            self.reset_state()
            return
        with self.lock:
            if self.temp_save is None:
                self.env.reset()
            else:
                self.env.load_state(self.temp_save)
        self.refresh_screen()
    def save_state(self,*args,**kwargs)->None:
        """Save the temp state."""
        with self.lock:
            self.temp_save=self.env.save_state()
    def change_gui_render(self,*args,**kwargs)->None:
        """Switches rendering mode."""
        with self.lock:
            self.env.change_gui_render()
            self.update_screen_shapes(True)
        self.refresh_screen()
    def follow_agent_path(self,*args,**kwargs)->None:
        """Follow movements from the agent prediction."""
        with self.lock:
            if len(self.pending_inputs)==0 and self.env.agent is not None:
                pred=self.env.agent.predict_action()
                self.pending_inputs=self.env.agent.scheduled_actions+[pred]
                self.env.agent.reset_scheduled_actions()
    def update_screen_content(self,screen:np.ndarray)->None:
        """Update displayed screen."""
        if self.quitting:
            return
        screen_scaled=screen if self.upscale<2 else np.repeat(np.repeat(screen,self.upscale,axis=0),self.upscale,axis=1)
        for y in range(0,screen_scaled.shape[0],screen_scaled.shape[0]//self.env.centered_screen_size[0]):
            screen_scaled[y,:]=0x3F
        for x in range(0,screen_scaled.shape[1],screen_scaled.shape[1]//self.env.centered_screen_size[1]):
            screen_scaled[:,x]=0x3F
        if len(screen_scaled.shape)<3:
            screen_scaled=np.expand_dims(screen_scaled,axis=-1)
        if screen_scaled.shape[-1]<3:
            screen_scaled=np.repeat(screen_scaled,3,axis=-1)
        self.img=ImageTk.PhotoImage(image=Image.fromarray(screen_scaled))
        self.canvas.itemconfig(self.canvas_image,image=self.img)
    def run(self,*args,**kwargs)->None:
        """Main loop."""
        self.window.after(1,lambda:self.step(True))
        self.window.mainloop()
    def run_quit(self,steps:int=-1)->None:
        """Main loop saving screenshots on quit."""
        self.screenshot_on_quit=self.env.log_screen
        self.env.max_steps=steps
        self.run()
        sys.exit(1)
