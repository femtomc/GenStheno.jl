module GenStheno

using Gen
using Stheno

# ------------ Trace ------------ #

struct SthenoTrace{N} <: Gen.Trace
    score::Float64
    gen_fn::GenerativeFunction
    args::Tuple
    retval::Array{Float64, N}
end

@inline Gen.get_args(trace::SthenoTrace) = trace.args
@inline Gen.get_retval(trace::SthenoTrace) = trace.retval
@inline Gen.get_score(trace::SthenoTrace) = trace.score
@inline Gen.get_choices(trace::SthenoTrace) = trace.retval
@inline Gen.get_gen_fn(trace::SthenoTrace) = trace.gen_fn

# ------------ Generative function ------------ #

struct SthenoGenerativeFunction <: GenerativeFunction
    model::Stheno.GP
    kernels::Vector
end

@inline (sf::SthenoGenerativeFunction)(args...) = Gen.get_retval(Gen.simulate(sf, args))

# ------------ GFI ------------ #

end # module
