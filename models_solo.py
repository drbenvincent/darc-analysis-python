import numpy as np
import pymc3 as pm
from theano import tensor as T
from df_plotting import *
from df_data import _data_df2dict, longest_delay

from sklearn.metrics import log_loss
import pandas as pd


def choice_func_psychometric(α, ϵ, VA, VB):
    """Calculate the probability of choosing option B.
    Used by PyMC3, therefore must be compatible with it."""
    return ϵ + (1.0-2.0*ϵ) * _cumulative_normal((VB-VA)/α)


def choice_func_psychometric2(α, ϵ, diff):
    """Calculate the probability of choosing option B.
    Used by PyMC3, therefore must be compatible with it."""
    return ϵ + (1.0-2.0*ϵ) * _cumulative_normal(diff/α)


def _cumulative_normal(x):
    """Calculates density of the cumulative standard normal distribution.
    Used by PyMC3, therefore must be compatible with it."""
    return 0.5 + 0.5 * pm.math.erf(x/pm.math.sqrt(2))


def calc_log_loss(R_predicted, R_actual):
    """Returns a distribution of log loss scores, one for each sample"""
    nsamples = R_predicted.shape[0]
    nresponses = R_actual.shape[0]
    assert np.ndim(R_predicted) == 2, "R_predicted is a vector(?) but should be a 2D matrix"
    assert R_predicted.shape[1] == nresponses, "cols in R_predicted should equal number of responses"
    print('Calculating Log Loss metric')
    R_actual = np.matlib.repmat(R_actual, nsamples, 1)
    ll = [log_loss(R_actual[n, :], R_predicted[n, :]) for n in range(0, nsamples)]
    return ll


class Ordered(pm.distributions.transforms.ElemwiseTransform):
    """Utility class for enforcing ordered variables in PyMC3 models"""
    name = "ordered"

    def forward(self, x):
        out = T.zeros(x.shape)
        out = T.inc_subtensor(out[0], x[0])
        out = T.inc_subtensor(out[1:], T.log(x[1:] - x[:-1]))
        return out

    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = T.zeros(y.shape)
        out = T.inc_subtensor(out[0], y[0])
        out = T.inc_subtensor(out[1:], T.exp(y[1:]))
        return T.cumsum(out)

    def jacobian_det(self, y):
        return T.sum(y[1:])


'''
                      _      _   _                           _
  _ __ ___   ___   __| | ___| | | |__   __ _ ___  ___    ___| | __ _ ___ ___
 | '_ ` _ \ / _ \ / _` |/ _ \ | | '_ \ / _` / __|/ _ \  / __| |/ _` / __/ __|
 | | | | | | (_) | (_| |  __/ | | |_) | (_| \__ \  __/ | (__| | (_| \__ \__ \
 |_| |_| |_|\___/ \__,_|\___|_| |_.__/ \__,_|___/\___|  \___|_|\__,_|___/___/

'''


class Model:
    """Base class for models"""
    # target_accept=.8
    model = None
    trace = None
    point_estimates = None

    def get_df_traces(self):
        """returns a dictionary containing the discount fraction related
        parameters"""
        return {key: self.trace[key] for key in self.df_params}

    def sample_posterior(self, data, nsamples=1000, tune=1000, target_accept=0.8):
        assert len(data) > 0, "No data!"
        model = self._build_model(data)
        print("sampling from model: ", self.__class__.__name__)
        with model:
            trace = pm.sample(nsamples, chains=4, tune=tune,
                              nuts_kwargs=dict(target_accept=target_accept))

        self.model = model
        self.trace = trace
        self.metrics = self._calc_metrics(data)
        self.point_estimates = self._calc_point_estimates()
        return

    def plot(self, data, ax, col='k'):
        print('ZZZZ Model base class method being called ZZZZ')
        pass


class BinaryResponseModel(Model):
    """Models where people are making binary choices between 2 prospects. This
    contrasts to matching designs where responses are continuous."""

    metrics = None

    def _calc_metrics(self, data):
        metrics = {}
        metrics['log_loss'] = calc_log_loss(self.trace.P_chooseB, data['R'])
        return metrics


class DiscountedUtilityModel(BinaryResponseModel):
    """Models based on discounted utility"""

    AUC = None

    def df_posterior_prediction(self, max_delay=365+7):
        assert self.trace is not None, "No trace found. Have you sampled yet?"
        delays = np.arange(0, max_delay, 1)
        param_dict = self.get_df_traces()
        discount_factor_matrix = self.df_plotting(delays, param_dict).T
        assert discount_factor_matrix.shape[0] == delays.shape[0], "Number of rows in posterior prediction should equal number of delays"
        return (delays, discount_factor_matrix)

    def plot(self, data, ax, col='k', plotCI=True, label=None, alpha=0.05):
        """Plot discount functions. Does the appropriate posterior prediction
        before"""

        # do posterior prediction
        max_delay = longest_delay(data)*1.1
        delays, df_pp_matrix = self.df_posterior_prediction(max_delay=max_delay)

        # plot CI region
        if plotCI:
            percentiles = 100 * np.array([alpha / 2., 1. - alpha / 2.])
            hpd = np.percentile(df_pp_matrix, percentiles, axis=1)
            ax.fill_between(delays, hpd[0], hpd[1], facecolor=col, alpha=0.25)

        # plot mean df
        curve_mean = df_pp_matrix.mean(axis=1)
        ax.plot(delays, curve_mean, color=col, label=self.__class__.__name__, lw=2.)

    def _calc_point_estimates(self):
        """ Store a dataframe of point estimates"""
        assert self.trace is not None, "No trace found. Have you sampled yet?"

        param_dict = self.get_df_traces()

        point_estimates = dict()
        for param in param_dict.keys():
            point_estimates[param] = np.median(param_dict[param])

        point_estimates['log loss'] = np.median(self.metrics['log_loss'])

        point_estimates['AUC'] = 'implement me'

        return pd.DataFrame(point_estimates, index=[0])


class HeuristicModel(BinaryResponseModel):
    """Heuristic models. No plotting (yet)"""

    def plot(self, data, ax, col='k'):
        pass

    def _calc_point_estimates(self):
        """ Store a dataframe of point estimates"""
        assert self.trace is not None, "No trace found. Have you sampled yet?"

        param_dict = self.get_df_traces()

        point_estimates = dict()
        for param in param_dict.keys():
            point_estimates[param] = np.median(param_dict[param])

        point_estimates['log loss'] = np.median(self.metrics['log_loss'])

        return pd.DataFrame(point_estimates, index=[0])

'''
                      _      _
  _ __ ___   ___   __| | ___| |___
 | '_ ` _ \ / _ \ / _` |/ _ \ / __|
 | | | | | | (_) | (_| |  __/ \__ \
 |_| |_| |_|\___/ \__,_|\___|_|___/

'''


class Coinflip(BinaryResponseModel):

    df_params = ['p']

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # having p separate from P_chooseB is intentional, to ensure trace
            # of P_chooseB is the same shape as the other models
            p = pm.Beta('p', alpha=1+1, beta=1+1)  # prior
            P_chooseB = pm.Deterministic('P_chooseB', p*np.ones(data['R'].shape))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood', p=P_chooseB, observed=data['R'])

        return model

    def plot(self, data, ax, col='k'):
        """No plotting for the coinflip model"""
        pass

    def _calc_point_estimates(self):
        """ Store a dataframe of point estimates"""
        assert self.trace is not None, "No trace found. Have you sampled yet?"

        param_dict = self.get_df_traces()

        point_estimates = dict()
        for param in param_dict.keys():
            point_estimates[param] = np.median(param_dict[param])

        point_estimates['log loss'] = np.median(self.metrics['log_loss'])

        return pd.DataFrame(point_estimates, index=[0])


class Exponential(DiscountedUtilityModel):
    """The classic Samuelson (1937) Exponential discount function"""

    df_params = ['k', 'alpha']

    @staticmethod
    def _df(k, delay):
        """Discount function, which must be compatable with PyMC3"""
        return pm.math.exp(-k * delay)

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays
        params = dictionary of numpy arrays
        """
        return np.exp(-np.outer(params['k'], delay))

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            k = pm.Bound(pm.Normal, lower=-0.005)('k', mu=0.001, sd=0.5)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(k, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(k, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB',
                                         choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class Hyperbolic(DiscountedUtilityModel):
    """Classic hyperboloid function of Mazur (1987)"""

    df_params = ['logk', 'alpha']

    @staticmethod
    def _df(logk, delay):
        """Discount function, which must be compatable with PyMC3"""
        k = pm.math.exp(logk)
        return 1. / (1.0+k*delay)

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays"""
        k = np.exp(params['logk'])
        return 1. / (1.0+np.outer(k, delay))

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            logk = pm.Normal('logk', mu=-4, sd=5)
            α = pm.Exponential('alpha', lam=1)
            # ϵ = pm.Beta('epsilon', alpha=1.1, beta=10.9)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(logk, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(logk, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class HyperbolicMagnitudeEffect(DiscountedUtilityModel):
    """Classic hyperboloid function of Mazur (1987), but where log discounte
    rate is linearly related to log reward magnitude. See Vincent (2016)."""

    df_params = ['m', 'c', 'alpha']

    @staticmethod
    def _df(m, c, delay, reward):
        """Discount function, which must be compatable with PyMC3"""
        logk = m*pm.math.log(reward)+c
        k = pm.math.exp(logk)
        return 1. / (1.0+k*delay)

    @staticmethod
    def df_plotting(delay, params):
        # TODO: impliment this
        pass

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            m = pm.Normal('m', mu=-0.243, sd=5)
            c = pm.Normal('c', mu=-4.7, sd=10)
            α = pm.Exponential('alpha', lam=1)
            # ϵ = pm.Beta('epsilon', alpha=1.1, beta=10.9)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(m, c, data['DA'], data['A']))
            VB = pm.Deterministic('VB', data['B'] * self._df(m, c, data['DB'], data['B']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model

    def plot(self, data):
        """No plotting of discount surfaces yet."""
        pass


class HyperboloidA(DiscountedUtilityModel):
    """Myerson & Green (1995) Hyperboloid"""

    df_params = ['logk', 's', 'alpha']

    @staticmethod
    def _df(logk, s, delay):
        """Discount function, which must be compatable with PyMC3"""
        k = pm.math.exp(logk)
        return 1.0 / (1.0+k*delay)**s

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays.
        Number of rows should equal number of delays"""
        k = np.exp(params['logk'])
        temp = k*delay[:, np.newaxis]
        return (1./((1.+temp)**params['s'])).T

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            logk = pm.Normal('logk', mu=-4, sd=5)
            s = pm.Bound(pm.Normal, lower=0)('s', mu=1, sd=2)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(logk, s, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(logk, s, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class HyperboloidB(DiscountedUtilityModel):
    """Rachlin (2006) Hyperboloid. I've also called this the Hyperbolic Power
     model"""

    df_params = ['logk', 's', 'alpha']

    @staticmethod
    def _df(logk, s, delay):
        """Discount function, which must be compatable with PyMC3"""
        k = pm.math.exp(logk)
        return 1.0 / (1.0+k*(delay**s))

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays.
        Number of rows should equal number of delays"""
        k = np.exp(params['logk'])
        temp = k*(delay[:, np.newaxis]**params['s'])
        return (1./(1.+temp)).T

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            logk = pm.Normal('logk', mu=-4, sd=5)
            s = pm.Bound(pm.Normal, lower=0)('s', mu=1, sd=2)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(logk, s, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(logk, s, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class HyperbolicLog(DiscountedUtilityModel):
    """Hyperbolic discounting of log scaled time.
    1/(1+k.log(1+S.D))"""

    df_params = ['logk', 's', 'alpha']

    @staticmethod
    def _df(logk, s, delay):
        """Discount function, which must be compatable with PyMC3"""
        k = pm.math.exp(logk)
        return 1.0 / (1.0 + k * pm.math.log(1+s*delay))

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays.
        Number of rows should equal number of delays"""
        delay = delay[np.newaxis, :]
        s = params['s']
        k = np.exp(params['logk'])
        s = s[:, np.newaxis]
        k = k[:, np.newaxis]
        return 1./(1. + k * np.log(1. + s * delay))

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            logk = pm.Normal('logk', mu=-4, sd=5)
            s = pm.Bound(pm.Normal, lower=0)('s', mu=1, sd=2)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(logk, s, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(logk, s, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class ConstantSensitivity(DiscountedUtilityModel):
    """Constant Sensitivity discount function by Ebert et al (2007)."""

    df_params = ['k', 's', 'alpha']

    @staticmethod
    def _df(k, s, delay):
        """Discount function, which must be compatable with PyMC3"""
        return pm.math.exp(-(k * delay)**s)

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays
        params = dictionary of numpy arrays
        """

        a = params['k'] * delay[:, np.newaxis]
        b = a**params['s']
        return np.exp(-b).T

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            k = pm.Bound(pm.Normal, lower=-0.005)('k', mu=0.001, sd=0.5)
            s = pm.Bound(pm.Normal, lower=0)('s', mu=1, sd=2)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(k, s, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(k, s, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB',
                                         choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class ExponentialPower(DiscountedUtilityModel):
    """Exponential Power model
    Similar in form to the Constant Sensitivity discount function."""

    df_params = ['k', 's', 'alpha']

    @staticmethod
    def _df(k, s, delay):
        """Discount function, which must be compatable with PyMC3"""
        return pm.math.exp(-(k * (delay**s)))

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays
        params = dictionary of numpy arrays
        """
        return np.exp(-params['k'] * (delay[:, np.newaxis]**params['s'])).T

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            k = pm.Bound(pm.Normal, lower=-0.005)('k', mu=0.001, sd=0.5)
            s = pm.Bound(pm.Normal, lower=0)('s', mu=1, sd=2)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(k, s, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(k, s, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB',
                                         choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class ExponentialLog(DiscountedUtilityModel):
    """Exponential Log model
    exp(-k.ln(1+S.D))"""

    df_params = ['logk', 's', 'alpha']

    @staticmethod
    def _df(logk, s, delay):
        """Discount function, which must be compatable with PyMC3"""
        k = np.exp(logk)
        return pm.math.exp(-k * pm.math.log(1+s*delay))

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays
        params = dictionary of numpy arrays
        """
        delay = delay[np.newaxis, :]
        s = params['s']
        k = np.exp(params['logk'])
        s = s[:, np.newaxis]
        k = k[:, np.newaxis]
        return np.exp(-k * np.log(1 + s * delay))

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            #k = pm.Normal('k', mu=0.001, sd=0.5)
            logk = pm.Normal('logk', mu=-4, sd=5)
            s = pm.Bound(pm.Normal, lower=0)('s', mu=1, sd=2)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(logk, s, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(logk, s, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB',
                                         choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class DoubleExponential(DiscountedUtilityModel):
    """Double Exponential discount function by McClure et al (2007)"""

    df_params = ['k', 'p', 'alpha']

    @staticmethod
    def _df(k, p, delay):
        """Discount function, which must be compatable with PyMC3.
        Core discount function is
            (p * exp(-k[0]*d)) + ((1-p) * exp(-k[1]*d))
        """
        return p * pm.math.exp(-k[0] * delay) + (1-p) * pm.math.exp(-k[1] * delay)

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays
        """
        k = params['k']
        a = np.exp(-np.outer(k[:, 0], delay))
        b = np.exp(-np.outer(k[:, 1], delay))
        return params['p'][:, np.newaxis] * a + (1-params['p'][:, np.newaxis]) * b

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            k = pm.Normal('k', mu=0.01, sd=1., shape=2, transform=Ordered(),
                          testval=[0.01, 0.02])
            p = pm.Beta('p', alpha=1+4, beta=1+4)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(k, p, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(k, p, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class BetaDelta(DiscountedUtilityModel):
    """Beta Delta discount function by Phelps & Pollak (1968)"""

    df_params = ['beta', 'delta', 'alpha']

    @staticmethod
    def _df(beta, delta, delay):
        """Discount function, which must be compatable with PyMC3"""
        # set values to 1 where delay==0
        return T.switch(delay == 0, 1.0, beta * (delta**delay))

    @staticmethod
    def df_plotting(delay, params):
        """Discount function, used for plotting.
        Has to be able to handle vector inputs for the parameters and delays"""
        temp = (params['beta'] * (params['delta']**delay[:, np.newaxis])).T
        # set values to 1 where delay==0
        temp[:,delay==0] = 1
        print(temp.shape)
        return temp

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors: β is intercept, δ is slope
            β = pm.Bound(pm.Normal, lower=0)('beta', mu=1, sd=1)
            δ = pm.Normal('delta', mu=1, sd=0.1)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01
            # Value functions
            VA = pm.Deterministic('VA', data['A'] * self._df(β, δ, data['DA']))
            VB = pm.Deterministic('VB', data['B'] * self._df(β, δ, data['DB']))
            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB', choice_func_psychometric(α, ϵ, VA, VB))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


'''
   _                     _     _   _                            _      _
  | |__   ___ _   _ _ __(_)___| |_(_) ___   _ __ ___   ___   __| | ___| |___
  | '_ \ / _ \ | | | '__| / __| __| |/ __| | '_ ` _ \ / _ \ / _` |/ _ \ / __|
  | | | |  __/ |_| | |  | \__ \ |_| | (__  | | | | | | (_) | (_| |  __/ \__ \
  |_| |_|\___|\__,_|_|  |_|___/\__|_|\___| |_| |_| |_|\___/ \__,_|\___|_|___/

These are in progress!

'''


class TradeOff(HeuristicModel):
    """Trade Off model by Scholten & Read (2010)
    Note that this is a Heuristic model, outside the category of discounted
    utility models.
    This follows the implementation in Ericson et al (2015) and the follow up
    critique of Wulff & van den Bos (2017)

    NOTE: CURRENTLY VALID FOR GAINS ONLY, I BELIEVE"""

    df_params = ['beta','gamma', 'tau']

    @staticmethod
    def _time_weighing_function(theta, x):
        """Helper function, which must be compatable with PyMC3.
        See their equation 10."""
        return pm.math.log(1.+theta*x)/theta

    @staticmethod
    def _value_function(theta, x):
        """Helper function, which must be compatable with PyMC3.
        NOTE: THIS IS VALID FOR GAINS ONLY! SEE THEIR EQUATION 9"""
        return pm.math.log(1.+theta*x)/theta

    @staticmethod
    def df_plotting(delay, params):
        pass

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            # NOTE: we need another variable if we deal with losses, which goes
            # to the value function
            β = pm.Bound(pm.Normal, lower=0)('beta', mu=1, sd=1000)
            γ = pm.Bound(pm.Normal, lower=0)('gamma', mu=0, sd=1000)
            τ = pm.Bound(pm.Normal, lower=0)('tau', mu=0, sd=1000)

            # TODO: pay attention to the choice function & it's params
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01

            value_diff = (self._value_function(γ, data['B'])
                          - self._value_function(γ, data['A']))
            time_diff = (self._time_weighing_function(τ, data['DB'])
                         - self._time_weighing_function(τ, data['DA']))
            diff = value_diff - β * time_diff

            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB',
                                         choice_func_psychometric2(α, ϵ, diff))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class ITCH(HeuristicModel):
    """ITCH model by Ericson et al (2015)
    Ericson et al (2015) and the follow up critique of Wulff & van den Bos (2017)
    """

    df_params = ['beta','alpha']

    @staticmethod
    def df_plotting(delay, params):
        pass

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            β = pm.Normal('beta', mu=0, sd=1, shape=5)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01

            A = data['B']-data['A']
            B = (data['B']-data['A'])/((data['B']+data['A'])/2)
            C = data['DB']-data['DA']
            D = (data['DB']-data['DA'])/((data['DB']+data['DA'])/2)
            diff = β[0] + β[1]*A + β[2]*B + β[3]*C + β[4]*D

            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB',
                                         choice_func_psychometric2(α, ϵ, diff))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model


class DRIFT(HeuristicModel):
    """DRIFT model by Read et al (2013)
    Based upon the implementation of Wulff & van den Bos (2017)
    Note that this is a Heuristic model, outside the category of discounted
    utility models.
    """

    df_params = ['beta', 'alpha']

    @staticmethod
    def df_plotting(delay, params):
        pass

    def _build_model(self, data):
        data = _data_df2dict(data)
        with pm.Model() as model:
            # Priors
            β = pm.Normal('beta', mu=0, sd=10, shape=4)
            α = pm.Exponential('alpha', lam=1)
            ϵ = 0.01

            D = data['B']-data['A']
            R = (data['B']-data['A'])/data['A']
            T = data['DB']-data['DA']
            I = ((data['B']/data['A'])**(1./(data['DB']-data['DA'])))-1.
            diff = β[0] + β[0]*D + β[1]*R + β[2]*I + β[3]*T

            # Choice function: psychometric
            P_chooseB = pm.Deterministic('P_chooseB',
                                         choice_func_psychometric2(α, ϵ, diff))
            # Likelihood of observations
            r_likelihood = pm.Bernoulli('r_likelihood',
                                        p=P_chooseB,
                                        observed=data['R'])

        return model
