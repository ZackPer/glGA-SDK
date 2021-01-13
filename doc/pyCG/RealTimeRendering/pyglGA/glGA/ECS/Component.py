"""
Component classes, part of the glGA SDK ECS
    
glGA SDK v2020.1 ECS (Entity Component System)
@Coopyright 2020 George Papagiannakis
    
The Compoment class is the dedicated to a specific type of data container in the glGA ECS.

The following is example restructured text doc example
:param file_loc: The file location of the spreadsheet
:type file_loc: str
:returns: a list of strings representing the header columns
:rtype: list

"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from System import *

class Component(ABC):
    """
    The abstract Component class of our ECS.
    
    Based on the Strategy pattern, it is a data collection of specific
    class of data. Subclasses typically are e.g. Transform, Mesh, Shader, RigidBody etc.
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
    
    def init(self):
        """
        abstract method to be subclassed for extra initialisation
        """
        pass
    
    def update(self):
        """
        method to be subclassed for debuging purposes only, 
        in case we need some behavioral or logic computation within te Component. 
        This violates the ECS architecture and should be avoided.
        """
        pass
    
    def accept(self, system: System):
        """
        Accepts a class object to operate on the Component, based on the Visitor pattern.

        :param system: [a System object]
        :type system: [System]
        """
        system.update()

class BasicTransform(Component):
    """
    An example of a concrete Component Transform class
    
    Contains a basic Euclidean Translation, Rotation and Scale Homogeneous Vector
    
    and a 4x4 matrix of all of them
    
    :param Component: [description]
    :type Component: [type]
    """
   
    def featureA(self):
        print(self.getClassName(),":featureA() called")
        
    def update(self):
        self.featureA()


class Mesh(Component):
    """
    A concrete Mesh class

    :param Component: [description]
    :type Component: [type]
    """
    def featureB(self):
        print(self.getClassName(), ":featureB() called")
        
    def update(self):
        self.featureB()
   
    