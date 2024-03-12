import gmsh

infile = r"pcb.step"
gmsh.initialize()
gmsh.option.setNumber("General.NumThreads", 4)
gmsh.option.setNumber("General.Terminal", 1)

#gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
#gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
#gmsh.option.setNumber("Mesh.MeshSizeMax", 10)
#gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 6)
#gmsh.option.setNumber("Mesh.RandomFactor", 0.10)
#gmsh.option.setNumber("Mesh.Algorithm", 8)


gmsh.model.add("model_1")

gmsh.merge(infile)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write(infile.replace(".step", "22.inp"))
gmsh.finalize()
