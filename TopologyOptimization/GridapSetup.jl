############## Gridap Setup ##################
model = GmshDiscreteModel("t2.msh")

order = 1
diritags = ["TopSide","BottomSide", "LeftSide", "RightSide", "OuterVertices"]
V = TestFESpace(
  reffe=:Lagrangian,
  order=1,
  valuetype=Float64,
  model=model,
  conformity=:H1,
  dirichlet_tags=diritags)

U = TrialFESpace(V,[0,0,0,0,0])

trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

############### Get the design region and target region ################
labels = get_face_labeling(model)
dimension = num_cell_dims(model)
tags = get_face_tag(labels,dimension)
const design_tag = get_tag_from_name(labels,"Design")
const target_tag = get_tag_from_name(labels,"Target")
cellmask_d = get_face_mask(labels,"Design",dimension)
cellmask_t = get_face_mask(labels,"Target",dimension)
# Subset of the mesh
trian_d = Triangulation(model,cellmask_d)
quad_d = CellQuadrature(trian_d,degree)
trian_t = Triangulation(model,cellmask_t)
quad_t = CellQuadrature(trian_t,degree)
# Number of cells in design region (number of design parameters)
np = num_cells(trian_d)

# Boundary of the design region
neumanntags = ["InnerVertices", "InnerSides"]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,degree)
######################## Design Parameter FE Space #####################
# Piece-wise constant FE space for the design parameters p (before filter and threshold)
Q = TestFESpace(
  reffe=:Lagrangian,
  order=0,
  valuetype=Float64,
  conformity=:L2,
  model=model)

P = TrialFESpace(Q)

# FE function space for the filtered parameters pf
Qf = TestFESpace(
  model=model,
  reffe=:Lagrangian,
  order=1,
  conformity=:H1,
  valuetype=Float64)
Pf = TrialFESpace(Qf)


################### Assemble matrix and vector #####################
# Material distribution
# pt = Threshold(Filter(p))
@law ξ(pt,α) = 1.0/(ϵ_1 + (ϵ_2-ϵ_1)*pt-α*1im*pt*(1-pt))
# Weak form of the Helmholtz equation : a(p,u,v)=Λ⋅∇v⋅ξ(p)⋅Λ⋅∇u-k²μv⋅u
a(u,v,pfh,α,β,η,k,σ,LH,d_pml) = ((x->Λ(x,σ,k,LH,d_pml))⋅∇(v))⊙(ξ(Threshold(β,η,pfh),α)*((x->Λ(x,σ,k,LH,d_pml))⋅∇(u))) - k^2*μ*v⊙u
# Source term (point source approximation)
f(x,I,δ,pos) = I/(2π*δ^2)*exp(-((x[1]-pos[1])^2+(x[2]-pos[2])^2)/δ^2/2)
b_Ω(v) = v*(x->f(x,I,δ,pos))

# Construct the finite element matrix and vector in Gridap
function GridapFEM(pf,b_Ω,α,β,η,k,σ,LH,d_pml)
    pfh  = FEFunction(Pf,pf)
    t_Ω = AffineFETerm((u,v)->a(u,v,pfh,α,β,η,k,σ,LH,d_pml),b_Ω,trian,quad)
    op = AffineFEOperator(SparseMatrixCSC{ComplexF64,Int},Vector{ComplexF64},U,V,t_Ω)
    A = get_matrix(op)
    b = get_vector(op)
    A, b
end