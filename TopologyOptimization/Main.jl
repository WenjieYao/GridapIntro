using Gmsh
using Gridap
using GridapGmsh
using SparseArrays
using Gridap.Geometry
using ChainRules
using Zygote
using NLopt
import Gmsh: gmsh
import ChainRules: rrule

Gridap.outer(a::Number,b::Number) = a * b
Gridap.Helpers.operate(::typeof(tanh),x::Float64)=tanh(x)

include("MeshGenerator.jl")
include("PML.jl")
include("Helper.jl")
include("FilterAndThreshold.jl")

# Physical parameters 
λ = 1.0          # Wavelength (arbitrary unit)
k = 2*π/λ        # Wave number 
ω = k            # c=1
ϵ_1 = 1.0        # Relative electric permittivity for material 1
ϵ_2 = 3.0        # Relative electric permittivity for material 2
μ = 1.0          # Relative magnetic permeability for all materials

# Geometry parameters of the mesh
L = 4.0          # Length of the normal region
H = 4.0          # Height of the normal region
d_pml = 0.8      # Thickness of the PML
L_d = L/2        # Length of the design region
H_d = H/5        # Height of the design region
r_t = L/40       # Radius of the target circle
y_t = -(H_d+H)/4 # y-position of the target circle (x fixed to 0)
LH = [L,H]

# Characteristic length (controls the resolution, smaller the finer)
resol = 10       # Number of points per wavelength
l_0 = λ/resol    # Normal region
l_d = l_0/5      # Design region
l_pml = 2*l_0    # PML 

# Point source location
pos = [0.0,H/2.0*0.9]
δ = λ/resol      # Gaussian point source width
I = 1e4          # Gaussian point source amplitude

# PML parameters
R = 1e-4         # Tolerence for PML reflection 
σ = -3/4*log(R)/d_pml/sqrt(ϵ_1)

# Generate mesh
MeshGenerator(L,H,L_d,H_d,r_t,y_t,d_pml,l_0,l_d,l_pml)
include("GridapSetup.jl")
include("Objective.jl")
# Filter and threshold paramters
r = l_d*1.0      # Filter radius
β = 5.0          # β∈[1,∞], threshold sharpness
η = 0.5          # η∈[0,1], threshold center

# Loss control
α = 0.           # α∈[0,∞], controls the material loss

opt = Opt(:LD_MMA, np)
opt.lower_bounds = 0.0
opt.upper_bounds = 1.0
opt.ftol_rel = 1e-3
opt.maxeval = 500
opt.max_objective = g_p

(g_opt,p_opt,ret) = optimize(opt, rand(np))
#(g_opt,p_opt,ret) = optimize(opt, p)
numevals = opt.numevals # the number of function evaluations

# Display u and ε
p = p_opt
pf = pf_p(p)
uvec = u_pf(pf)
ϵ0 = ϵ_1 + (ϵ_2-ϵ_1)*FEFunction(P,p_vec(p,P,tags,design_tag))
ϵt = ϵ_1 + (ϵ_2-ϵ_1)*Threshold(β,η,FEFunction(Pf,pf))
writevtk(trian,"demo",cellfields=["ϵ0"=>ϵ0,"ϵt"=>ϵt,"Real"=>FEFunction(U,real(uvec)),"imag"=>FEFunction(U,imag(uvec)),"Norm"=>FEFunction(U,sqrt.(real(uvec).^2+imag(uvec).^2))])

