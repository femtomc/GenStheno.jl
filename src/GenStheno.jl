module GenStheno

using Gen
using Stheno

# ------------ Choice map ------------ #

struct SthenoChoiceMap{N} <: ChoiceMap
    val::Array{Float64, N}
end

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
@inline Gen.get_choices(trace::SthenoTrace) = SthenoChoiceMap(trace.retval)
@inline Gen.get_gen_fn(trace::SthenoTrace) = trace.gen_fn

# ------------ Generative function ------------ #

struct SthenoGenerativeFunction <: GenerativeFunction
    generator::Function
    σ²_n::Float64 # iid observation noise
end

@inline (sf::SthenoGenerativeFunction)(args...) = Gen.get_retval(Gen.simulate(sf, args))

# ------------ GFI ------------ #

function Gen.simulate(gen_fn::SthenoGenerativeFunction, args::Tuple)
    gp = gen_fn.generator(args...)
    @assert gp isa Stheno.GP
    fx = gp(x, gen_fn.σ²_n)
    s = rand(fx)
    score = logpdf(fx, s)
    SthenoTrace(score, gen_fn, args, s)
end

function Gen.generate(gen_fn::SthenoGenerativeFunction, args::Tuple, chm::SthenoChoiceMap)
    gp = gen_fn.generator(args...)
    @assert gp isa Stheno.GP
    constraint = chm.val
    fx = gp(x, gen_fn.σ²_n)
    weight = lmle(gp, constraint)
    SthenoTrace(weight, gen_fn, args, constraint), weight
end

function Gen.propose(gen_fn::SthenoGenerativeFunction, args::Tuple)
    trace = simulate(gen_fn, args)
    retval = get_retval(trace)
    (SthenoChoiceMap(retval), get_score(trace), retval)
end

function Gen.update(trace::SthenoTrace, args::Tuple, argdiffs::Tuple, chm::SthenoChoiceMap)
    new_trace, w = generate(trace.gen_fn, args, chm)
    (new_trace, w, UnknownChange(), get_choices(trace))
end

function Gen.update(trace::SthenoTrace, args::Tuple, argdiffs::Tuple, chm::EmptyChoiceMap)
    all(x -> x isa NoChange, argdiffs) && return (trace, 0., NoChange(), EmptyChoiceMap())
    new_trace = simulate(trace.gen_fn, args)
    (new_trace, get_score(trace) - get_score(new_trace), UnknownChange(), EmptyChoiceMap())
end

function Gen.regenerate(trace::SthenoTrace, args::Tuple, argdiffs::Tuple,::EmptySelection)
    all(x -> x isa NoChange, argdiffs) && return (trace, 0.0, NoChange())
    new_trace = simulate(trace.gen_fn, args)
    (new_tr, get_score(new_trace) - get_score(trace), UnknownChange())
end

function Gen.regenerate(trace::SthenoTrace, args::Tuple, argdiffs::Tuple,::SelectAll)
    new_trace = simulate(trace.gen_fn, args)
    (new_tr, get_score(new_trace) - get_score(trace), UnknownChange())
end

# ------------ Macro ------------ #

function _genstheno(expr) end
macro sthen(expr)
    new = _genstheno(expr)
    esc(new)
end

end # module
