"""
System classes, part of the glGA SDK ECS
    
glGA SDK v2020.1 ECS (Entity Component System)
@Coopyright 2020 George Papagiannakis
    
The System class is the logic-specific processor of different Components in the glGA ECS.

The following is example restructured text doc example
:param file_loc: The file location of the spreadsheet
:type file_loc: str
:returns: a list of strings representing the header columns
:rtype: list

"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from Component import *
from Entity import *

class System(ABC):
    """
    Main abstract class of the System part of our ECS
    
    Typically involves all logic involving operations such as Rendering, Local2World, Physics

    :param ABC: [description]
    :type ABC: [type]
    """
    
    def __init__(self, name=None, type=None, id=None):
        self._name = name
        self._type = type
        self._id = id
    
    #define properties for id, name, type
     
    @property #name
    def name(self) -> str:
        """ Get Component's name """
        return self._name
    @name.setter
    def name(self, value):
        self._name = value
        
    @property #type
    def type(self) -> str:
        """ Get Component's type """
        return self._type
    @type.setter
    def type(self, value):
        self._type = value
        
    @property #id
    def id(self) -> str:
        """ Get Component's id """
        return self._id
    @id.setter
    def id(self, value):
        self._id = value
    
    @classmethod
    def getClassName(cls):
        return cls.__name__
    
    def update(self):
        """
        method to be subclassed for  behavioral or logic computation 
        when visits Components of an EntityNode. 
        
        """
        pass
    
    def apply(self, Entity):
        """
        method to be subclassed for  behavioral or logic computation 
        when visits Entities. 
        
        """
        pass
    
    
    def apply(self, renderMesh: RenderMesh):
        """
        method to be subclassed for  behavioral or logic computation 
        when visits Components. 
        
        """
        pass
    
    
    def apply(self, basicTransform: BasicTransform):
        """
        method to be subclassed for  behavioral or logic computation 
        when visits Components. 
        
        """
        pass


class TransformUpdate(System):
    """

    :param System: [description]
    :type System: [type]
    :return: [description]
    :rtype: [type]
    """
    
    def update(self):
        """
        method to be subclassed for  behavioral or logic computation 
        when visits Components of an EntityNode. 
        
        """
        pass
    
    
    def apply(self, basicTransform: BasicTransform):
        """
        method to be subclassed for  behavioral or logic computation 
        when visits Components. 
        
        """
        print(self.getClassName(), ": apply(BasicTransform) called")
        basicTransform.update()


class RenderGPU(System):
    """
    A basic forward rendering system based on GPU shaders
    :param System: [description]
    :type System: [type]
    """
    
    def update(self):
        """
        method to be subclassed for  behavioral or logic computation 
        when visits Components of an EntityNode. 
        
        """
        pass
    
    
    def apply(self, renderMesh: RenderMesh):
        """
        method to be subclassed for  behavioral or logic computation 
        when visits Components. 
        
        """
        print(self.getClassName(), ": apply(RenderMesh) called")
        renderMesh.update()
    
    
            
