using Gridap
using SparseArrays
using SpecialFunctions

# Geometry parameters
L = 4.0          # Length of the square area
d_pml = 0.8      # Thickness of the PML
resolution = 20  # Number of cells per wavelength

# Physical parameters
λ = 1.0          # Wavelength (arbitrary unit)
k = 2*π/λ        # Wave number 
δ = λ/resolution # Gaussian point source width

# PML parameters
R = 1e-4         # Tolerence for PML reflection 
σ = -3/4*log(R)/d_pml

# PML coordinate stretching functions
function s_PML(x,σ,k,L,d_pml)
    u = abs.(Tuple(x)).-L./2
    return @. ifelse(u > 0,  1-(1im*σ/k)*(u/d_pml)^2, $(1.0+0im))
end

function Λ(x,σ,k,L,d_pml)  
    s_x,s_y = s_PML(x,σ,k,L,d_pml)
    return TensorValue(1/s_x,0,0,1/s_y) 
end

# Create Cartesian mesh and the model
# (L+2d_pml)*(L+2d_pml) domain size with the center (0,0)
domain = (-L/2-d_pml,L/2+d_pml,-L/2-d_pml,L/2+d_pml)
Nc = round(Int,(L+2d_pml)/λ*resolution)     # Number of cells per side
cells = (Nc,Nc)
model = CartesianDiscreteModel(domain,cells)

# Generate triangulation and quadrature from model 
trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

# Test and trial finite element function space
order = 1
V = TestFESpace(
  reffe=:Lagrangian,
  order=1,
  valuetype=Float64,
  model=model,
  conformity=:H1,
  dirichlet_tags="boundary")

U = TrialFESpace(V,0)


# Define the theoretic model
# Bilinear term
a(u,v,k,σ,L,d_pml) = ((x->Λ(x,σ,k,L,d_pml))⋅∇(v))⊙((x->Λ(x,σ,k,L,d_pml))⋅∇(u)) - k^2*v⊙u
# Source term (Gaussian point source approximation at center)
f(x,δ) = 1/(2π*δ^2)*exp(-(norm(x)/δ)^2/2)
b_Ω(v,δ) = v*(x->f(x,δ))

# Assemble the matrix and solve for the field
t_Ω = AffineFETerm((u,v)->a(u,v,k,σ,L,d_pml),v->b_Ω(v,δ),trian,quad)
op = AffineFEOperator(SparseMatrixCSC{ComplexF64,Int},Vector{ComplexF64},U,V,t_Ω)
A = get_matrix(op)
b = get_vector(op)
uvec = A\b

# Analytical solution for a magnetic dipole
function H_t(x,λ)
    kr = 2*π/λ*norm(x)
    return -0.25im*hankelh2(0,kr)
end

b_theory(v) = v*(x->H_t(x,λ))
t_theory = AffineFETerm((u,v)->(u*v),b_theory,trian,quad)
op_t = AffineFEOperator(SparseMatrixCSC{ComplexF64,Int},Vector{ComplexF64},U,V,t_theory)
A_t = get_matrix(op_t)
b_t = get_vector(op_t)
u_t = A_t\b_t

# Compare the relative difference
# Since there will always be a large difference at the center 
# and in PML, we use x/(1+x) to show the difference
Gridap.outer(a::Number,b::Number) = a*b
@law myabs(x) = abs(x)/(1+abs(x))
diff = myabs(FEFunction(U,uvec)-FEFunction(U,u_t))

# Save to file and view
writevtk(trian,"demo",cellfields=["Real"=>FEFunction(U,real(uvec)),
        "Imag"=>FEFunction(U,imag(uvec)),
        "Norm"=>FEFunction(U,sqrt.(real(uvec).^2+imag(uvec).^2)),
        "Real_t"=>FEFunction(U,real(u_t)),
        "Imag_t"=>FEFunction(U,imag(u_t)),
        "Norm_t"=>FEFunction(U,sqrt.(real(u_t).^2+imag(u_t).^2)),
        "Difference"=>diff])