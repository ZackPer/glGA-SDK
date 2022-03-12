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
from enum import Enum
from random import uniform;
import numpy as np
import imgui
import sys;
sys.path.append("C:\\Users\\User\\Documents\\glGA-SDK\\packages");

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
 
class imgui_GameObjectProperties(ImGUIDecorator):
    # This UI should provide a better tool for UI
    def __init__(self, wrapee: RenderWindow, imguiContext = None):
        super().__init__(wrapee, imguiContext);
        # Name
        self.name = "Random comp";
        # TRS
        self.translation    = [0, 0, 0, 0];
        self.rotation       = [0, 0, 0, 0];
        self.scale          = [0, 0, 0, 0];
        # Material
        
    def Draw(self):
        imgui.begin("Inspector: " + self.name);
        imgui.end();
        None;

    None;

class ImGUIecssDecorator(ImGUIDecorator):
    """custom ImGUI decorator for this example

    :param ImGUIDecorator: [description]
    :type ImGUIDecorator: [type]
    """
    def __init__(self, wrapee: RenderWindow, imguiContext = None):
        super().__init__(wrapee, imguiContext)
        # @GPTODO:
        # we should be able to retrieve all these just from the Scene: ECSSManager
        self.translation = [0.0, 0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0, 0.0];
        self.mvpMat = None
        self.shaderDec = None
        
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
            self.drawNode(self.wrapeeWindow.scene.world.root, self.translation, self.rotation)
            imgui.tree_pop()
        imgui.next_column()
        imgui.text("Properties")
        imgui.separator()
        #TRS sample
        changed, trans = imgui.drag_float4("Translation", *self.translation)
        self.translation = list(trans)
        changed, rot = imgui.drag_float4("Rotation", *self.rotation)
        self.rotation = list(rot)
        
        imgui.end()
        
    def drawNode(self, component, translation = None, rotation = None):
        #save initial translation value
        lastTranslation = translation
        lastRotation = rotation;
        
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
                            print(f'Selected: {selected} of node: {comp}'.center(100, '-'))
                            selected = False
                            #check if the component is a BasicTransform
                            if (isinstance(comp, BasicTransform)):
                                #set now the comp:
                                comp.trs = util.translate(lastTranslation[0],lastTranslation[1],lastTranslation[2])
                                #retrive the translation vector from the TRS matrix
                                # @GPTODO this needs to be provided as utility method
                                trsMat = comp.trs
                                [x,y,z] = trsMat[:3,3]
                                if translation is not None:
                                    translation[0] = x
                                    translation[1] = y
                                    translation[2] = z
                                    translation[3] = 1
                                # Same for rotation
                                # comp.trs = trns
                                # set now the comp:
                                
                                # Rotation test
                                rotateMat = util.rotate((1.0, 0.0, 0.0), lastRotation[0]) @ util.rotate((0.0, 1.0, 0.0), lastRotation[1]) @ util.rotate((0.0, 0.0, 1.0), lastRotation[2])
                                comp.trs = comp.trs @ rotateMat;
                                #retrive the translation vector from the TRS matrix
                                # @GPTODO this needs to be provided as utility method
                                trsMat = comp.trs
                                rotation[0] = lastRotation[0];
                                rotation[1] = lastRotation[1];
                                rotation[2] = lastRotation[2];

                        imgui.tree_pop()
                    
                    self.drawNode(comp, translation, rotation) # recursive call of this method to traverse hierarchy
                    imgui.unindent(10) # Corrent placement of unindent

class GameObjectEntity(Entity):
    def __init__(self, name=None, type=None, id=None) -> None:
        super().__init__(name, type, id);

        # Create basic components of a primitive object
        self.trans          = BasicTransform(name="trans", trs=util.identity());
        self.mesh           = RenderMesh(name="mesh");
        self.shaderDec      = ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source = Shader.COLOR_FRAG));
        self.vArray         = VertexArray();
        # Add components to entity
        scene = Scene();
        scene.world.createEntity(self);
        scene.world.addComponent(self, self.trans);
        scene.world.addComponent(self, self.mesh);
        scene.world.addComponent(self, self.shaderDec);
        scene.world.addComponent(self, self.vArray);

    def SetVertexAttributes(self, vertex, color, index):
        self.mesh.vertex_attributes.append(vertex);
        self.mesh.vertex_attributes.append(color);
        self.mesh.vertex_index.append(index);

    def SetColor(self, r, g, b):
        color = self.mesh.vertex_attributes[1];
        for vertex in color:
            vertex[0] = r;
            vertex[1] = b;
            vertex[2] = g;

class PrimitiveGameObjectType(Enum):
    CUBE = 0
    PYRAMID = 1

class PrimitiveGameObjectSpawner():
    _instance = None;
    __dispatcher = {};

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PrimitiveGameObjectSpawner, cls).__new__(cls);
            cls._instance.__initialize();
        return cls._instance;
    def __initialize(self):
        def CubeSpawn(): 
            cube = GameObjectEntity("Cube");
            vertexCube = np.array(
                [
                    [-0.5, -0.5, 0.5, 1.0],
                    [-0.5, 0.5, 0.5, 1.0],
                    [0.5, 0.5, 0.5, 1.0],
                    [0.5, -0.5, 0.5, 1.0], 
                    [-0.5, -0.5, -0.5, 1.0], 
                    [-0.5, 0.5, -0.5, 1.0], 
                    [0.5, 0.5, -0.5, 1.0], 
                    [0.5, -0.5, -0.5, 1.0]
                ],
                dtype=np.float32
            ) 
            colorCube = np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0]
                ], 
                dtype=np.float32
            )
            #index arrays for above vertex Arrays
            indexCube = np.array(
                (
                    1,0,3, 1,3,2, 
                    2,3,7, 2,7,6,
                    3,0,4, 3,4,7,
                    6,5,1, 6,1,2,
                    4,5,6, 4,6,7,
                    5,4,0, 5,0,1
                ),
                np.uint32
            ) #rhombus out of two triangles
            cube.SetVertexAttributes(vertexCube, colorCube, indexCube);

            return cube;
        def PyramidSpawn():
            pyramid = GameObjectEntity("Pyramid");
            vertexPyramid = np.array(
                [
                    [-0.5, -0.5, -0.5, 1.0],
                    [-0.5, -0.5, 0.5, 1.0],
                    [0.5, -0.5, 0.5, 1.0],
                    [0.5, -0.5, -0.5, 1.0],
                    [0.0, 0.5, 0.0, 1.0],
                ],
                dtype=np.float32
            ) 
            colorPyramid = np.array(
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                ], 
                dtype=np.float32
            )
            #index arrays for above vertex Arrays
            indexPyramid = np.array(
                (
                    1,0,3, 1,3,2,
                    0,3,4,
                    0,1,4,
                    1,4,2,
                    2,4,3
                ),
                np.uint32
            ) #rhombus out of two pyramids
            pyramid.SetVertexAttributes(vertexPyramid, colorPyramid, indexPyramid);
            return pyramid;

        self.__dispatcher[PrimitiveGameObjectType.CUBE] = CubeSpawn;
        self.__dispatcher[PrimitiveGameObjectType.PYRAMID] = PyramidSpawn;
        None;
    
    def Spawn(self, type: PrimitiveGameObjectType):
        return self.__dispatcher[type]();
        None;

def SpawnHome():
    scene = Scene();

    home = scene.world.createEntity(Entity("Home"));
    scene.world.addEntityChild(scene.world.root, home);

    # Add trs to home
    trans = BasicTransform(name="trans", trs=util.identity());    scene.world.addComponent(home, trans);

    # Create simple pyramid
    pyramid: GameObjectEntity = PrimitiveGameObjectSpawner().Spawn(PrimitiveGameObjectType.PYRAMID);
    scene.world.addEntityChild(home, pyramid);
    pyramid.trans.trs = util.translate(0, 1, 0); # Move pyramid to the top of the cube
    # Create simple cube
    cube: GameObjectEntity = PrimitiveGameObjectSpawner().Spawn(PrimitiveGameObjectType.CUBE);
    scene.world.addEntityChild(home, cube);

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
    entityCam1 = scene.world.createEntity(Entity(name="entityCam1"))
    scene.world.addEntityChild(rootEntity, entityCam1)
    trans1 = scene.world.addComponent(entityCam1, BasicTransform(name="trans1", trs=util.identity()))
    
    entityCam2 = scene.world.createEntity(Entity(name="entityCam2"))
    scene.world.addEntityChild(entityCam1, entityCam2)
    trans2 = scene.world.addComponent(entityCam2, BasicTransform(name="trans2", trs=util.identity()))
    # orthoCam = scene.world.addComponent(entityCam2, Camera(util.ortho(-5.0, 5.0, -5.0, 5.0, 0.0, 100.0), "orthoCam","Camera","500"))
    orthoCam = scene.world.addComponent(entityCam2, Camera(util.perspective(20, 1, 0, 100), "orthoCam","Camera","500"))

    home1: Entity = SpawnHome();
    home1.getChild(0).trs = util.translate(0, 0, 0) @ util.scale(0.5, 0.5, 0.5);
    home1.getChild(1).SetColor(255, 255, 0);

    home2: Entity = SpawnHome();
    home2.getChild(0).trs = util.translate(1, 0, 0) @ util.scale(0.5, 0.5, 0.5);

    applyUniformTransformList = [];
    applyUniformTransformList.append(home1);
    applyUniformTransformList.append(home2);

    # Camera settings
    trans2.trs = util.translate(0, 0, 8) # VIEW
    trans1.trs = util.identity(); # MODEL
        
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
        scene.world.traverse_visit_pre_camera(camUpdate, orthoCam)
        # 3. run proper Ml2c traversal
        scene.world.traverse_visit(camUpdate, scene.world.root)
        
        # 3.1 shader uniform variable allocation per frame
        for uniformItem in applyUniformTransformList:
            for child in uniformItem._children:
                if(isinstance(child, GameObjectEntity)):
                    child.shaderDec.setUniformVariable(key='modelViewProj', value=child.trans.l2cam, mat4=True)

        
        # 4. call SDLWindow/ImGUI display() and ImGUI event input process
        running = scene.render(running)
        # 5. call the GL State render System
        scene.world.traverse_visit(renderUpdate, scene.world.root)
        # 6. ImGUI post-display calls and SDLWindow swap 
        scene.render_post()
        
    scene.shutdown()


if __name__ == "__main__":    
    main(imguiFlag = True)