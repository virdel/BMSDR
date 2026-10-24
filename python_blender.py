import bpy
import sys


data_path=str(sys.argv[-2])
output_path=str(sys.argv[-1])

print("Using blender uv generating....")
print("input path:",data_path)
print("out path:",output_path)

# data_path = "E:\mesh_50_600.obj"
for obj in bpy.data.objects:   # 迭代列出bpy.data.objects中的所有物体
    print(obj)

bpy.data.objects.remove(bpy.data.objects['Cube'])
bpy.ops.outliner.orphans_purge(do_recursive=True)

bpy.ops.import_scene.obj(filepath=data_path)

# bpy.ops.object.select_all(action='DESELECT')
# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.editmode_toggle()
# bpy.ops.mesh.select_all(action='SELECT') # for all faces
# bpy.ops.uv.smart_project(angle_limit=66, island_margin = 0.02)
# bpy.ops.object.editmode_toggle()

context = bpy.context
scene = context.scene
vl = context.view_layer
# deselect all to make sure select one at a time
bpy.ops.object.select_all(action='DESELECT')

for obj in scene.objects:
    if (obj.type == 'MESH'):
        vl.objects.active = obj
        obj.select_set(True)
        print(obj.name)
        lm = obj.data.uv_layers.get("LightMap")
        if not lm:
            lm = obj.data.uv_layers.new(name="LightMap")
        lm.active = True
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action='SELECT')  # for all faces
        # bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
        bpy.ops.uv.smart_project()
        # bpy.ops.uvpackeroperator.packbtn()
        bpy.ops.object.editmode_toggle()
        obj.select_set(False)


bpy.ops.export_scene.obj(filepath=output_path)


