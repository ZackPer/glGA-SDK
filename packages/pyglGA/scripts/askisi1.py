"""
BasicWindow example, showcasing the pyglGA SDK ECSS
    
glGA SDK v2021.0.5 ECSS (Entity Component System in a Scenegraph)
@Coopyright 2020-2021 George Papagiannakis
    
The classes below are all related to the GUI and Display of 3D 
content using the OpenGL, GLSL and SDL2, ImGUI APIs, on top of the
pyglGA ECSS package
"""

from __future__         import annotations
from asyncore import dispatcher
from math import sin, cos, radians
from enum import Enum
from random import uniform;
import numpy as np
import imgui
import sys

from zmq import ROUTER_MANDATORY

from pyglGA.scripts.IndexedConverter import IndexedConverter;
sys.path.append("C:\\Users\\User\\Documents\\glGA-SDK\\packages");

import OpenGL.GL as gl;
import pyglGA.ECSS.utilities as util
from pyglGA.ECSS.System import System, TransformSystem, CameraSystem
from pyglGA.ECSS.Entity import Entity
from pyglGA.ECSS.Component import BasicTransform, Camera, RenderMesh
from pyglGA.ECSS.Event import Event, EventManager
from pyglGA.GUI.Viewer import SDL2Window, ImGUIDecorator, RenderGLStateSystem, RenderWindow
from pyglGA.ECSS.ECSSManager import ECSSManager
from pyglGA.ext.Shader import InitGLShaderSystem, Shader, ShaderGLDecorator, RenderGLShaderSystem
from pyglGA.ext.VertexArray import VertexArray
from pyglGA.ext.Scene import Scene
 
class ImGUIecssDecorator(ImGUIDecorator):
    """custom ImGUI decorator for this example

    :param ImGUIDecorator: [description]
    :type ImGUIDecorator: [type]
    """
    def __init__(self, wrapee: RenderWindow, imguiContext = None):
        super().__init__(wrapee, imguiContext)
        self.selected = None; # Selected should be a component

        # TRS Variables 
        self.translation = {};
        self.translation["x"] = 0; self.translation["y"] = 0; self.translation["z"] = 0; 

        self.rotation = {};
        self.rotation["x"] = 0; self.rotation["y"] = 0; self.rotation["z"] = 0; 

        self.scale = {};
        self.scale["x"] = 0; self.scale["y"] = 0; self.scale["z"] = 0; 

        self.color = [255, 50, 50];
        
    def scenegraphVisualiser(self):
        """display the ECSS in an ImGUI tree node structure
        Typically this is a custom widget to be extended in an ImGUIDecorator subclass 
        """
        sceneRoot = self.wrapeeWindow.scene.world.root.name
        if sceneRoot is None:
            sceneRoot = "ECSS Root Entity"
        
        imgui.begin("ECSS graph")
        imgui.columns(2,"Properties")
        # below is a recursive call to build-up the whole scenegraph as ImGUI tree
        if imgui.tree_node(sceneRoot, imgui.TREE_NODE_OPEN_ON_ARROW):
            self.drawNode(self.wrapeeWindow.scene.world.root)
            imgui.tree_pop()
        imgui.next_column()
        imgui.text("Properties")
        imgui.separator()

        #TRS sample
        # if(isinstance(self.selected, BasicTransform)):
        if imgui.tree_node("Translation", imgui.TREE_NODE_OPEN_ON_ARROW):
            changed, value = imgui.slider_float("X", self.translation["x"], -3, 3, "%.01f", 1);
            self.translation["x"] = value;
            changed, value = imgui.slider_float("Y", self.translation["y"], -3, 3, "%.01f", 1);
            self.translation["y"] = value;
            changed, value = imgui.slider_float("Z", self.translation["z"], -3, 3, "%.01f", 1);
            self.translation["z"] = value;
            imgui.tree_pop();
        if imgui.tree_node("Rotation", imgui.TREE_NODE_OPEN_ON_ARROW):
            changed, value = imgui.slider_float("X", self.rotation["x"], -90, 90, "%.1f", 1);
            self.rotation["x"] = value;
            changed, value = imgui.slider_float("Y", self.rotation["y"], -90, 90, "%.1f", 1);
            self.rotation["y"] = value;
            changed, value = imgui.slider_float("Z", self.rotation["z"], -90, 90, "%.1f", 1);
            self.rotation["z"] = value;
            imgui.tree_pop();
        if imgui.tree_node("Scale", imgui.TREE_NODE_OPEN_ON_ARROW):
            changed, value = imgui.slider_float("X", self.scale["x"], 0, 3, "%.01f", 1);
            self.scale["x"] = value;
            changed, value = imgui.slider_float("Y", self.scale["y"], 0, 3, "%.01f", 1);
            self.scale["y"] = value;
            changed, value = imgui.slider_float("Z", self.scale["z"], 0, 3, "%.01f", 1);
            self.scale["z"] = value;
            imgui.tree_pop();

        imgui.end()
        
    def drawNode(self, component):
        #create a local iterator of Entity's children
        if component._children is not None:
            debugIterator = iter(component._children)
            #call print() on all children (Concrete Components or Entities) while there are more children to traverse
            done_traversing = False
            while not done_traversing:
                try:
                    comp = next(debugIterator)
                    imgui.indent(10)
                except StopIteration:
                    done_traversing = True
                    # imgui.unindent(10) # Wrong placement of uinindent
                else:
                    if imgui.tree_node(comp.name + " | " + str(comp.id), imgui.TREE_NODE_OPEN_ON_ARROW):
                        #imgui.text(comp.__str__())
                        _, selected = imgui.selectable(comp.__str__(), True)
                        if selected:
                            if comp != self.selected: # First time selecting it. Set trs values to GUI;
                                self.selected = comp;
                                if isinstance(comp, BasicTransform):
                                    [x, y, z] = comp.translation;
                                    self.translation["x"] = x;
                                    self.translation["y"] = y;
                                    self.translation["z"] = z;
                                    [x, y, z] = comp.scale;
                                    self.scale["x"] = x;
                                    self.scale["y"] = y;
                                    self.scale["z"] = z;
                                    [x, y, z] = comp.rotationEulerAngles;
                                    self.rotation["x"] = x;
                                    self.rotation["y"] = y;
                                    self.rotation["z"] = z;
                                elif isinstance(comp, GameObjectEntity):
                                    self.color = comp.color.copy();
                            else:                       # Set GUI values to trs;
                                if isinstance(comp, BasicTransform):
                                    transMat = util.translate(self.translation["x"], self.translation["y"], self.translation["z"]);
                                    rotMatX = util.rotate((1, 0, 0), self.rotation["x"])
                                    rotMatY = util.rotate((0, 1, 0), self.rotation["y"])
                                    rotMatZ = util.rotate((0, 0, 1), self.rotation["z"])
                                    scaleMat = util.scale(self.scale["x"], self.scale["y"], self.scale["z"])

                                    comp.trs = util.identity() @ transMat @ rotMatX @ rotMatY @ rotMatZ @ scaleMat;
                                    # comp.trs = scaleMat @ rotMatZ @ rotMatY @ rotMatX @ transMat;
                                elif hasattr(comp, "drawSelfGui"):
                                    comp.drawSelfGui(imgui);

                        imgui.tree_pop()
                    
                    self.drawNode(comp) # recursive call of this method to traverse hierarchy
                    imgui.unindent(10) # Corrent placement of unindent

class GameObjectEntity(Entity):
    def __init__(self, name=None, type=None, id=None) -> None:
        super().__init__(name, type, id);

        # Gameobject basic properties
        self._color          = [1, 1, 1]; # this will be used as a uniform var
        # Create basic components of a primitive object
        self.trans          = BasicTransform(name="trans", trs=util.identity());
        self.mesh           = RenderMesh(name="mesh");
        self.shaderDec      = ShaderGLDecorator(Shader(vertex_source=Shader.VERT_PHONG_MVP, fragment_source=Shader.FRAG_PHONG));
        self.vArray         = VertexArray();
        # Add components to entity
        scene = Scene();
        scene.world.createEntity(self);
        scene.world.addComponent(self, self.trans);
        scene.world.addComponent(self, self.mesh);
        scene.world.addComponent(self, self.shaderDec);
        scene.world.addComponent(self, self.vArray);

    @property
    def color(self):
        return self._color;
    @color.setter
    def color(self, colorArray):
        self._color = colorArray;

    def drawSelfGui(self, imgui):
        changed, value = imgui.color_edit3("Color", self.color[0], self.color[1], self.color[2]);
        self.color = [value[0], value[1], value[2]];

    def SetVertexAttributes(self, vertex, color, index, normals = None):
        self.mesh.vertex_attributes.append(vertex);
        self.mesh.vertex_attributes.append(color);
        if normals is not None:
            self.mesh.vertex_attributes.append(normals);
        self.mesh.vertex_index.append(index);

class Light(Entity):
    def __init__(self, name=None, type=None, id=None) -> None:
        super().__init__(name, type, id);
        # Add variables for light
        self.color = [1, 1, 1];
        self.intensity = 1;
    
    def drawSelfGui(self, imgui):
        changed, value = imgui.slider_float("Intensity", self.intensity, 0, 10, "%.1f", 1);
        self.intensity = value;

        changed, value = imgui.color_edit3("Color", self.color[0], self.color[1], self.color[2]);
        self.color = [value[0], value[1], value[2]];
        None;

class PointLight(Light):
    def __init__(self, name=None, type=None, id=None) -> None:
        super().__init__(name, type, id);

        # Create basic components of a primitive object
        self.trans          = BasicTransform(name="trans", trs=util.identity());
        scene = Scene();
        scene.world.createEntity(self);
        scene.world.addComponent(self, self.trans);

    def drawSelfGui(self, imgui):
        super().drawSelfGui(imgui);

class SimpleCamera(Entity):
    def __init__(self, name=None, type=None, id=None) -> None:
        super().__init__(name, type, id)
        scene = Scene();
        rootEntity = scene.world.root;

        scene.world.addEntityChild(rootEntity, self);

        entityCam1 = scene.world.createEntity(Entity(name="entityCam1"));
        scene.world.addEntityChild(self, entityCam1);
        self.trans1 = scene.world.addComponent(entityCam1, BasicTransform(name="trans1", trs=util.identity()));
        
        entityCam2 = scene.world.createEntity(Entity(name="entityCam2"));
        scene.world.addEntityChild(entityCam1, entityCam2);
        self.trans2 = scene.world.addComponent(entityCam2, BasicTransform(name="trans2", trs=util.identity()));
        
        self._near = 1;
        self._far = 20;

        self._fov = 50;
        self._aspect = 1.0;

        self._left = -10;
        self._right = 10;
        self._bottom = -10;
        self._top = 10;

        self._mode = "perspective";
        self._camera = scene.world.addComponent(entityCam2, Camera(util.perspective(self._fov, self._aspect, self._near, self._far), "MainCamera", "Camera", "500"));        
        None;

    @property
    def camera(self):
        return self._camera;

    def drawSelfGui(self, imgui):
        if imgui.button("Orthograpic") and self._mode == "perspective":
            self._mode = "orthographic";
            self._camera.projMat = util.ortho(self._left, self._right, self._bottom, self._top, self._near, self._far);
        if imgui.button("Perspective") and self._mode == "orthographic":
            self._mode = "perspective";
            self._camera.projMat = util.perspective(self._fov, self._aspect, self._near, self._far)

        if self._mode == "orthographic":
            changed, value = imgui.slider_float("Left", self._left, -50, -1, "%0.1f", 1);
            self._left = value;
            changed, value = imgui.slider_float("Right", self._right, 1, 50, "%0.1f", 1);
            self._right = value;
            changed, value = imgui.slider_float("Bottom", self._bottom, -50, -1, "%0.1f", 1);
            self._bottom = value;
            changed, value = imgui.slider_float("Top", self._top, 1, 50, "%0.1f", 1);
            self._top = value;

            self._camera.projMat = util.ortho(self._left, self._right, self._bottom, self._top, self._near, self._far);
        elif self._mode == "perspective":
            changed, value = imgui.slider_float("FOV", self._fov, 1, 120, "%0.1f", 1);
            self._fov = value;
            changed, value = imgui.slider_float("Aspect", self._aspect, 0.5, 2, "%0.1f", 1);
            self._aspect = value;

            self._camera.projMat = util.perspective(self._fov, self._aspect, self._near, self._far)

class PrimitiveGameObjectType(Enum):
    CUBE = 0
    PYRAMID = 1
    QUAD = 2

class PrimitiveGameObjectSpawner():
    _instance = None;
    __dispatcher = {};

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PrimitiveGameObjectSpawner, cls).__new__(cls);
            cls._instance.__initialize();
        return cls._instance;
    def __initialize(self):
        def QuadSpawn():
            quad = GameObjectEntity("Quad");
            vertices = np.array(
                [
                    [-1, 0, -1, 1.0],
                    [1, 0, -1, 1.0], 
                    [-1, 0, 1, 1.0],
                    [1, 0, 1, 1.0],
                ],
                dtype=np.float32
            )
            colors = np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32
            )
            indices = np.array(
                (
                    1, 0, 3,
                    2, 3, 0
                ),
                np.uint32
            )
            normals = [];
            for i in range(0, len(indices), 3):
                normals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
                normals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
                normals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
                None;
            quad.SetVertexAttributes(vertices, colors, indices, normals);
            return quad;
        def CubeSpawn(): 
            cube = GameObjectEntity("Cube");
            vertices = [
                [-0.5, -0.5, 0.5, 1.0],
                [-0.5, 0.5, 0.5, 1.0],
                [0.5, 0.5, 0.5, 1.0],
                [0.5, -0.5, 0.5, 1.0], 
                [-0.5, -0.5, -0.5, 1.0], 
                [-0.5, 0.5, -0.5, 1.0], 
                [0.5, 0.5, -0.5, 1.0], 
                [0.5, -0.5, -0.5, 1.0]
            ];
            colors = [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0]                    
            ];
            #index arrays for above vertex Arrays
            indices = np.array(
                (
                    1,0,3, 1,3,2, 
                    2,3,7, 2,7,6,
                    3,0,4, 3,4,7,
                    6,5,1, 6,1,2,
                    4,5,6, 4,6,7,
                    5,4,0, 5,0,1
                ),
                dtype=np.uint32
            ) #rhombus out of two triangles

            vertices, colors, normals, indices = IndexedConverter().Convert(vertices, colors, indices, produceNormals=True);
            cube.SetVertexAttributes(vertices, colors, indices, normals);
            
            return cube;
        def PyramidSpawn():
            pyramid = GameObjectEntity("Pyramid");
            vertices = [
                [-0.5, -0.5, -0.5, 1.0],
                [-0.5, -0.5, 0.5, 1.0],
                [0.5, -0.5, 0.5, 1.0],
                [0.5, -0.5, -0.5, 1.0],
                [0.0, 0.5, 0.0, 1.0],
            ]; 
            colors = [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ];
            #index arrays for above vertex Arrays
            indices = np.array(
                (
                    1,0,3, 1,3,2,
                    3,0,4,
                    0,1,4,
                    1,2,4,
                    2,3,4
                ),
                np.uint32
            ) #rhombus out of two pyramids
            
            vertices, colors, normals, indices = IndexedConverter().Convert(vertices, colors, indices, produceNormals=True);
            pyramid.SetVertexAttributes(vertices, colors, indices, normals);

            return pyramid;

        self.__dispatcher[PrimitiveGameObjectType.CUBE] = CubeSpawn;
        self.__dispatcher[PrimitiveGameObjectType.PYRAMID] = PyramidSpawn;
        self.__dispatcher[PrimitiveGameObjectType.QUAD] = QuadSpawn;
        None;
    
    def Spawn(self, type: PrimitiveGameObjectType):
        return self.__dispatcher[type]();

class RotateAnimation(Entity):
    def __init__(self, name=None, type=None, id=None) -> None:
        super().__init__(name, type, id);
        self._angle = 1;
        self._target = None;

    def SetTarget(self, target: BasicTransform):
        self._target = target;

    def Progress(self):
        self._target.trs = self._target.trs @ util.rotate((0, 1, 0), self._angle);

    def drawSelfGui(self, imgui):
        changed, value = imgui.slider_float("Euler Angle", self._angle, -20, 20, "%.1f", 1);
        self._angle = value;
    

def SpawnHome():
    scene = Scene();

    home = scene.world.createEntity(Entity("Home"));
    scene.world.addEntityChild(scene.world.root, home);

    # Add trs to home
    trans = BasicTransform(name="trans", trs=util.identity());    scene.world.addComponent(home, trans);

    # Create simple cube
    cube: GameObjectEntity = PrimitiveGameObjectSpawner().Spawn(PrimitiveGameObjectType.CUBE);
    scene.world.addEntityChild(home, cube);

    # Create simple pyramid
    pyramid: GameObjectEntity = PrimitiveGameObjectSpawner().Spawn(PrimitiveGameObjectType.PYRAMID);
    scene.world.addEntityChild(home, pyramid);
    pyramid.trans.trs = util.translate(0, 1, 0); # Move pyramid to the top of the cube

    return home;


def main(imguiFlag = False):
    ##########################################################
    # Instantiate a simple complete ECSS with Entities, 
    # Components, Camera, Shader, VertexArray and RenderMesh
    #
    #########################################################
    """
    ECSS for this example:
    
    root
        |---------------------------|           
        entityCam1,                 node4,      
        |-------|                    |--------------|----------|--------------|           
        trans1, entityCam2           trans4,        mesh4,     shaderDec4     vArray4
                |              applyCamera2BasicTransform                 
                ortho, trans2                   
                                                            
    """
    scene = Scene()    

    # Initialize Systems used for this script
    transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
    camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
    renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
    initUpdate = scene.world.createSystem(InitGLShaderSystem())
    
    # Scenegraph with Entities, Components
    rootEntity = scene.world.createEntity(Entity(name="Root"))

    # Spawn Camera
    mainCamera = SimpleCamera("Simple Camera");

    #  Spawn light
    ambientLight = Light("Ambient Light");
    ambientLight.intensity = 0.1;
    scene.world.addEntityChild(rootEntity, ambientLight);
    pointLight = PointLight();
    pointLight.trans.trs = util.translate(0.8, 1, 1)
    scene.world.addEntityChild(rootEntity, pointLight);

    # Spawn homes
    home1: Entity = SpawnHome();
    home1.getChild(0).trs = util.translate(0, 0, 0);
    
    home2: Entity = SpawnHome();
    home2.getChild(0).trs = util.translate(2, 0, 2);

    # Spawn and apply animation to pyramid
    rotatingPyramid = PrimitiveGameObjectSpawner().Spawn(PrimitiveGameObjectType.PYRAMID);
    rotatingPyramid.trans.trs = util.translate(-2, 0, -1);
    scene.world.addEntityChild(rootEntity, rotatingPyramid);
    rotateAnimation = RotateAnimation("Rotate Animation");
    rotateAnimation.SetTarget(rotatingPyramid.trans);
    scene.world.addEntityChild(rotatingPyramid, rotateAnimation);

    applyUniformTransformList = [];
    applyUniformTransformList.append(home1.getChild(1));
    applyUniformTransformList.append(home1.getChild(2));
    applyUniformTransformList.append(home2.getChild(1));
    applyUniformTransformList.append(home2.getChild(2));
    applyUniformTransformList.append(rotatingPyramid);

    # Camera settings
    mainCamera.trans2.trs = util.translate(0, 0, 8) # VIEW
    mainCamera.trans1.trs = util.rotate((1, 0, 0), -40);
        
    scene.world.print()
    # scene.world.eventManager.print()
    
    
    # MAIN RENDERING LOOP
    running = True
    scene.init(imgui=True, windowWidth = 1024, windowHeight = 1024, windowTitle = "pyglGA Cube ECSS Scene", customImGUIdecorator = ImGUIecssDecorator)
    imGUIecss = scene.gContext


    # ---------------------------------------------------------
    #   Run pre render GLInit traversal for once!
    #   pre-pass scenegraph to initialise all GL context dependent geometry, shader classes
    #   needs an active GL context
    # ---------------------------------------------------------
    
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glDisable(gl.GL_CULL_FACE);

    # gl.glDepthMask(gl.GL_FALSE);  
    gl.glEnable(gl.GL_DEPTH_TEST);
    gl.glDepthFunc(gl.GL_LESS);
    scene.world.traverse_visit(initUpdate, scene.world.root)
    
    
    ############################################
    # Instantiate all Event-related key objects
    ############################################
    
    # instantiate new EventManager
    # need to pass that instance to all event publishers e.g. ImGUIDecorator
    eManager = scene.world.eventManager
    gWindow = scene.renderWindow
    gGUI = scene.gContext
    
    #simple Event actuator System
    renderGLEventActuator = RenderGLStateSystem()
    
    #setup Events and add them to the EventManager
    updateTRS = Event(name="OnUpdateTRS", id=100, value=None)
    updateBackground = Event(name="OnUpdateBackground", id=200, value=None)
    #updateWireframe = Event(name="OnUpdateWireframe", id=201, value=None)
    eManager._events[updateTRS.name] = updateTRS
    eManager._events[updateBackground.name] = updateBackground
    #eManager._events[updateWireframe.name] = updateWireframe # this is added inside ImGUIDecorator
    
    # Add RenderWindow to the EventManager subscribers
    # @GPTODO
    # values of these Dicts below should be List items, not objects only 
    #   use subscribe(), publish(), actuate() methhods
    #
    eManager._subscribers[updateTRS.name] = gGUI
    eManager._subscribers[updateBackground.name] = gGUI
    # this is a special case below:
    # this event is published in ImGUIDecorator and the subscriber is SDLWindow
    eManager._subscribers['OnUpdateWireframe'] = gWindow
    eManager._actuators['OnUpdateWireframe'] = renderGLEventActuator
    
    # MANOS - START
    eManager._subscribers['OnUpdateCamera'] = gWindow
    eManager._actuators['OnUpdateCamera'] = renderGLEventActuator
    # MANOS - END

    # Add RenderWindow to the EventManager publishers
    eManager._publishers[updateBackground.name] = gGUI


    while running:
        # ---------------------------------------------------------
        # run Systems in the scenegraph
        # root node is accessed via ECSSManagerObject.root property
        # normally these are run within the rendering loop (except 4th GLInit  System)
        # --------------------------------------------------------
        # 1. L2W traversal
        scene.world.traverse_visit(transUpdate, scene.world.root) 
        # 2. pre-camera Mr2c traversal
        scene.world.traverse_visit_pre_camera(camUpdate, mainCamera.camera)
        # 3. run proper Ml2c traversal
        scene.world.traverse_visit(camUpdate, scene.world.root)
        

        viewPos = mainCamera.trans2.l2world[:3, 3].tolist();
        lightPos = pointLight.trans.l2world[:3, 3].tolist();
        # 3.1 shader uniform variable allocation per frame
        for object in applyUniformTransformList:
            if(isinstance(object, GameObjectEntity)):
                object.shaderDec.setUniformVariable(key='modelViewProj', value=object.trans.l2cam, mat4=True);
                object.shaderDec.setUniformVariable(key='model', value=object.trans.l2world, mat4=True);

                object.shaderDec.setUniformVariable(key='ambientColor', value=ambientLight.color, float3=True);
                object.shaderDec.setUniformVariable(key='ambientStr', value=ambientLight.intensity, float1=True);
                object.shaderDec.setUniformVariable(key='shininess', value=0.5, float1=True);
                object.shaderDec.setUniformVariable(key='matColor', value=object.color, float3=True);

                object.shaderDec.setUniformVariable(key='viewPos', value=viewPos, float3=True);
                object.shaderDec.setUniformVariable(key='lightPos', value=lightPos, float3=True);
                object.shaderDec.setUniformVariable(key='lightColor', value=np.array(pointLight.color), float3=True);
                object.shaderDec.setUniformVariable(key='lightIntensity', value=pointLight.intensity, float1=True);

        # 3.2 progress animations
        rotateAnimation.Progress();

        # 4. call SDLWindow/ImGUI display() and ImGUI event input process
        running = scene.render(running)
        # 5. call the GL State render System
        scene.world.traverse_visit(renderUpdate, scene.world.root)
        # 6. ImGUI post-display calls and SDLWindow swap 
        scene.render_post()
        
    scene.shutdown()


if __name__ == "__main__":    
    main(imguiFlag = True)