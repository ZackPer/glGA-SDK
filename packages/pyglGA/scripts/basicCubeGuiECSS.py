"""
BasicWindow example, showcasing the pyglGA SDK ECSS
    
glGA SDK v2021.0.5 ECSS (Entity Component System in a Scenegraph)
@Coopyright 2020-2021 George Papagiannakis
    
The classes below are all related to the GUI and Display of 3D 
content using the OpenGL, GLSL and SDL2, ImGUI APIs, on top of the
pyglGA ECSS package
"""

from __future__         import annotations
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
        self.camOrthoLRBT = [-100.0, 100.0, -100.0, 100.0]
        self.camOrthoNF = [1.0, 100.0]
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
            self.drawNode(self.wrapeeWindow.scene.world.root, self.translation, self.camOrthoLRBT, self.camOrthoNF)
            imgui.tree_pop()
        imgui.next_column()
        imgui.text("Properties")
        imgui.separator()
        #TRS sample
        changed, trans = imgui.drag_float4("Translation", *self.translation)
        self.translation = list(trans)
        changedLRBT, orthoLRBT = imgui.drag_float4("Camera LRBT", *self.camOrthoLRBT)
        self.camOrthoLRBT = list(orthoLRBT)
        changedNF, orthoNF = imgui.drag_float2("Camera Near, Far", *self.camOrthoNF)
        self.camOrthoNF = list(orthoNF)
        
        imgui.end()
        
    def drawNode(self, component, translation = None, camLRBT = None, camNF = None):
        #save initial translation value
        lastTranslation = translation
        lastLRBT = camLRBT
        lastNF = camNF
        
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
                    imgui.unindent(10)
                else:
                    if imgui.tree_node(comp.name, imgui.TREE_NODE_OPEN_ON_ARROW):
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
                            if (isinstance(comp, Camera)):
                                print(comp, " ready to assign new camera values!")
                                #set now camera params
                                # @GPTODO
                                # very dirty code, this is for a proof of concept only, should be replaced!
                                comp.projMat = util.ortho(lastLRBT[0], lastLRBT[1],lastLRBT[2],lastLRBT[3],lastNF[0], lastNF[1])
                                self.mvpMat = comp.projMat 
                                if self.shaderDec is not None:
                                    self.shaderDec.setUniformVariable(key='modelViewProj', value=self.mvpMat, mat4=True)
                        
                        imgui.tree_pop()
                    self.drawNode(comp, translation, camLRBT, camNF) # recursive call of this method to traverse hierarchy
            
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
                |                               
                ortho, trans2                   
                                                            
    """
        
    scene = Scene()    
    
    # Scenegraph with Entities, Components
    rootEntity = scene.world.createEntity(Entity(name="Root"))
    entityCam1 = scene.world.createEntity(Entity(name="entityCam1"))
    scene.world.addEntityChild(rootEntity, entityCam1)
    trans1 = scene.world.addComponent(entityCam1, BasicTransform(name="trans1", trs=util.identity()))
    
    entityCam2 = scene.world.createEntity(Entity(name="entityCam2"))
    scene.world.addEntityChild(entityCam1, entityCam2)
    trans2 = scene.world.addComponent(entityCam2, BasicTransform(name="trans2", trs=util.identity()))
    orthoCam = scene.world.addComponent(entityCam2, Camera(util.ortho(-100.0, 100.0, -100.0, 100.0, 1.0, 100.0), "orthoCam","Camera","500"))
    
    node4 = scene.world.createEntity(Entity(name="node4"))
    scene.world.addEntityChild(rootEntity, node4)
    trans4 = scene.world.addComponent(node4, BasicTransform(name="tras4", trs=util.identity()))
    mesh4 = scene.world.addComponent(node4, RenderMesh(name="mesh4"))
    # decorated components and systems with sample, default pass-through shader with uniform MVP
    shaderDec4 = scene.world.addComponent(node4, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))
    
    #Simple Cube
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
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0]
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
    
    # 
    # MVP matrix calculation - 
    # now set directly here only for Testing
    # otherwise automatically picked up at ECSS VertexArray level from the Scenegraph System
    # same process as VertexArray is automatically populated from RenderMesh
    #
    model = util.translate(0.0,0.0,0.0)
    # model = util.translate(0.0,0.0,0.0)
    eye = util.vec(0.0, 0.0, -10.0)
    target = util.vec(0,0,0)
    up = util.vec(0.0, 1.0, 0.0)
    view = util.identity();
    # view = util.lookat(eye, target, up)
    #projMat = util.frustum(-10.0, 10.0,-10.0,10.0, -1.0, 10)
    #projMat = util.perspective(120.0, 1.33, 0.1, 100.0)
    #projMat = util.ortho(-100.0, 100.0, -100.0, 100.0, 1.0, 100.0)
    projMat = util.ortho(-5.0, 5.0, -5.0, 5.0, -1.0, 5.0)
    mvpMat = model @ view @ projMat
    # mvpMat =  projMat @ view @ model
    
    #
    # setup ECSS nodes pre-systems
    #
    orthoCam.projMat = projMat
    trans2.trs = view
    trans1.trs = model
    #l2cMat = node4.l2cam
        
    
    # attach a simple cube in a RenderMesh so that VertexArray can pick it up
    mesh4.vertex_attributes.append(vertexCube)
    mesh4.vertex_attributes.append(colorCube)
    mesh4.vertex_index.append(indexCube)
    vArray4 = scene.world.addComponent(node4, VertexArray())
    
    # Systems
    transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
    camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
    renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
    initUpdate = scene.world.createSystem(InitGLShaderSystem())


    scene.world.print()
    scene.world.eventManager.print()
    
   
    # MAIN RENDERING LOOP
    running = True
    scene.init(imgui=True, windowWidth = 1024, windowHeight = 768, windowTitle = "pyglGA Cube ECSS Scene", customImGUIdecorator = ImGUIecssDecorator)
    imGUIecss = scene.gContext
    imGUIecss.mvpMat = mvpMat
    imGUIecss.shaderDec = shaderDec4
    

    # ---------------------------------------------------------
    #   Run pre render GLInit traversal for once!
    #   pre-pass scenegraph to initialise all GL context dependent geometry, shader classes
    #   needs an active GL context
    # ---------------------------------------------------------
    scene.world.traverse_visit(initUpdate, scene.world.root)
    
    #
    # setup ECSS nodes after-systems
    #
    l2cMat = trans4.l2cam
    print(f'\nl2cMat: \n{l2cMat}')
    print(f'\nmvpMat: \n{mvpMat}')
    print(trans4)
    
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
        #shaderDec4.setUniformVariable(key='modelViewProj', value=l2cMat, mat4=True)
        # direct uniform variable shader setup
        # should be called before ImGUI and before drawing Geometry
        # shaderDec4.setUniformVariable(key='modelViewProj', value=gWindow._myCamera, mat4=True)
        # shaderDec4.setUniformVariable(key='modelViewProj', value=l2cMat, mat4=True)
        shaderDec4.setUniformVariable(key='modelViewProj', value=trans4.l2cam, mat4=True)
        
        # 4. call SDLWindow/ImGUI display() and ImGUI event input process
        running = scene.render(running)
        # 5. call the GL State render System
        scene.world.traverse_visit(renderUpdate, scene.world.root)
        # 6. ImGUI post-display calls and SDLWindow swap 
        scene.render_post()
        
    scene.shutdown()


if __name__ == "__main__":    
    main(imguiFlag = True)