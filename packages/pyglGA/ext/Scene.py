"""
Scene classes, part of the glGA SDK ECSS
    
glGA SDK v2021.0.5 ECSS (Entity Component System in a Scenegraph)
@Coopyright 2020-2021 George Papagiannakis
    
The Compoment class is the dedicated to a specific type of data container in the glGA ECS.

The following is example restructured text doc example
:param file_loc: The file location of the spreadsheet
:type file_loc: str
:returns: a list of strings representing the header columns
:rtype: list

"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np

import pyglGA.ECSS.utilities as util
from pyglGA.ECSS.Entity import Entity, EntityDfsIterator
from pyglGA.ECSS.Component import BasicTransform, Camera
from pyglGA.ECSS.System import System, TransformSystem, CameraSystem, RenderSystem
from pyglGA.ECSS.ECSSManager import ECSSManager
from pyglGA.GUI.Viewer import SDL2Window, ImGUIDecorator

class Scene():
    """
    Singleton Scene that assembles ECSSManager and Viewer classes together for Scene authoring
    in pyglGA. It also brings together the new extensions to pyglGA: Shader, VertexArray and 
    RenderMeshDecorators
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            print('Creating Scene Singleton Object')
            cls._instance = super(Scene, cls).__new__(cls)
            cls._renderWindow = None
            cls._gContext = None
            cls._world = ECSSManager() #which also instantiates an EventManager
            # add further init here
        return cls._instance
    
    
    def __init__(self):
        None;
    
    @property
    def renderWindow(self):
        return self._renderWindow
    
    @property
    def gContext(self):
        return self._gContext
    
    @property
    def world(self):
        return self._world
    
    
    def init(self, sdl2 = True, imgui = False, windowWidth = None, windowHeight = None, windowTitle = None, customImGUIdecorator = None):
        """call the init() of all systems attached to this Scene based on the Visitor pattern
        """
        #init Viewer GUI subsystem with just SDL2 window or also an ImGUI decorators
        if sdl2 == True:
            #create a basic SDL2 RenderWindow with a reference to the Scene and thus ECSSManager and EventManager
            self._renderWindow = SDL2Window(windowWidth, windowHeight, windowTitle, self)
            self._gContext = self._renderWindow
        
        if imgui == True and customImGUIdecorator == None:
            gGUI = ImGUIDecorator(self._renderWindow)
            self._gContext = gGUI
        elif imgui == True and customImGUIdecorator is not None:
            gGUI = customImGUIdecorator(self._renderWindow)
            self._gContext = gGUI
    
        print("mark 1");
        self._gContext.init()
        print("mark 2");
        self._gContext.init_post()
    
    
    def update(self):
        """call the update() of all systems attached to this Scene based on the Visitor pattern
        """
        pass
    
    
    def processInput(self):
        """process the user input per frame based on Strategy and Decorator patterns
        """
        pass
    
        
    def render(self, running:bool = True) ->bool :
        """call the render() of all systems attached to this Scene based on the Visitor pattern
        """
        self._gContext.display()
        still_runnning = self._gContext.event_input_process(running)
        
        return still_runnning
    
    def render_post(self):
        self._gContext.display_post()
    
    def run(self):
        """main loop Scene method based on the "gameloop" game programming pattern
        """
        pass
    
    
    def shutdown(self):
        """main shutdown Scene method based on the "gameloop" game programming pattern
        """
        self._gContext.shutdown()


if __name__ == "__main__":
    # The client singleton code.

    s1 = Scene()
    s2 = Scene()

    if id(s1) == id(s2):
        print("Singleton works, both Scenes contain the same instance.")
    else:
        print("Singleton failed, Scenes contain different instances.")
        