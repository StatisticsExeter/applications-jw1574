#| echo: false
#| eval: true
#| results: 'asis'
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'regression'


def _fit_modelQ(df):
    model = smf.mixedlm(
        "shortfall ~ n_rooms + I(n_rooms**2) + age", 
        df,
        groups=df["local_authority_code"]
    )
    result = model.fit()
    return result


def _save_model_summary(model, outpath):
    with open(outpath, "w") as f:
        f.write(model.summary().as_text())


def _random_effects(results):
    re_df = pd.DataFrame(results.random_effects).T
    re_df.columns = ['Intercept'] + [f"Slope_{i}" for i in range(len(re_df.columns)-1)]
    re_df['group'] = re_df.index

    stderr = np.sqrt(results.cov_re.iloc[0, 0])
    re_df['lower'] = re_df['Intercept'] - 1.96 * stderr
    re_df['upper'] = re_df['Intercept'] + 1.96 * stderr

    re_df = re_df.sort_values('Intercept')
    return re_df


def fit_model_Q():
    base_dir = find_project_root()

    # This is the correct vignettes directory
    vignettes_dir = base_dir / 'data_cache' / 'vignettes' / 'regression'
    vignettes_dir.mkdir(parents=True, exist_ok=True)

    models_dir = base_dir / 'data_cache' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')

    results = _fit_modelQ(df)

    # Save summary
    outpath = vignettes_dir / 'model2_fit.txt'
    _save_model_summary(results, outpath)

    # Save random effects
    _random_effects(results).to_csv(models_dir / 'reffs2.csv')


# Run it
fit_model_Q()
