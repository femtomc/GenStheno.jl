module Simple

using Gen
using Stheno
include("../src/GenStheno.jl")
using .GenStheno

gp_gen_func = @sthen (generator = () -> GP(Matern32(), GPC()), σ = 0.5)

x = [rand() for _ in 1 : 10]
tr = simulate(gp_gen_func, (x, ))
chm = get_choices(tr)
display(chm)
tr, w = generate(gp_gen_func, (x, ), chm)
chm = get_choices(tr)
display(chm)

end # module
