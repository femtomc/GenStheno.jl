module KernelStructureInference

include("../src/GenStheno.jl")
using .GenStheno
using Gen

include("shared.jl")
include("involution_mh.jl")
include("involution_mh_tree.jl")
include("lightweight.jl")

end # module
