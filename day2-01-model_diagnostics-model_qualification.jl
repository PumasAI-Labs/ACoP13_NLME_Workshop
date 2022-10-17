# Load the necessary Libraries
using Pumas
using PumasUtilities
using PharmaDatasets
using CSV

# Read in data
# iv infusion given over 2 hours with demographic information (age, weight, sex, crcl)
pkfile = dataset("nlme_sample.csv", String)
pkdata = CSV.read(pkfile, DataFrame; missingstrings=["NA", ""])

pop = read_pumas(
    pkdata,
    id=:ID,
    time=:TIME,
    amt=:AMT,
    covariates=[:WT, :AGE, :SEX, :CRCL, :GROUP],
    observations=[:DV],
    cmt=:CMT,
    evid=:EVID,
    rate=:RATE,
)


# 2-cpt with weight and CRCL
model = @model begin
    @param begin
        tvcl ∈ RealDomain(lower=0)
        tvvc ∈ RealDomain(lower=0)
        tvq ∈ RealDomain(lower=0)
        tvvp ∈ RealDomain(lower=0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(lower=0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates WT CRCL

    @pre begin
        wtCL = (WT / 70)^0.75
        wtV = (WT / 70)
        crcl_eff = (CRCL / 95)^0.75
        CL = tvcl * crcl_eff * wtCL * exp(η[1])
        Vc = tvvc * wtV * exp(η[2])
        Q = tvq
        Vp = tvvp
    end

    @dynamics Central1Periph1

    @derived begin
        CONC := @. Central / Vc
        DV ~ @. Normal(CONC, σ)
    end
end

# Parameter values
params =
    (tvvc=5, tvcl=0.02, tvq=0.01, tvvp=10, Ω=Diagonal([0.01, 0.01]), σ=0.01)

# Fit models
fit_results = fit(model, pop, params, Pumas.FOCE())
fit_results_naive = fit(model, pop, params, Pumas.NaivePooled(); omegas=(:Ω,))
fit_results_fixed = fit(model, pop, params, Pumas.FOCE(); constantcoef=(tvcl=0.3,))

# Confidence Intervals
fit_infer = infer(fit_results)
coeftable(fit_infer)  # DataFrame

# Confidence Intervals using bootstrap
fit_infer_bs = infer(fit_results, Pumas.Bootstrap(samples=100))
coeftable(fit_infer_bs)

# Confidence Intervals using SIR
fit_infer_sir = infer(fit_results, Pumas.SIR(samples=10, resamples=10))
coeftable(fit_infer_sir)

# Inspect the models
fit_inspect = inspect(fit_results)
fit_inspect_naive = inspect(fit_results_naive)
fit_inspect_fixed = inspec

# Loglikelihood and NONMEM's OFV with constant

# AIC

# BIC

# ϵ-Shinkrage

# η-Shinkrage

# Model metrics