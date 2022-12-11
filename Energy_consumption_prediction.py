{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/SusanSagwa/Energy-Consumption-Prediction-In-Smart-Buildings-Using-Ensemble-Learning/blob/main/Energy_consumption_prediction.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "s5H7cD1ML8h3"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import gc\n",
        "import numpy as np # linear algebra\n",
        "import pandas as pd # data processing, CSV file I/O\n",
        "import datetime\n",
        "from sklearn.model_selection import RandomizedSearchCV, GridSearchCV\n",
        "from sklearn.tree import DecisionTreeRegressor\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.svm import SVR\n",
        "from sklearn.ensemble import GradientBoostingRegressor\n",
        "from sklearn.ensemble import StackingRegressor\n",
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "vFdlcON40wRF",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e0cfa658-994f-4df5-dd58-68b5c93a4bb0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "Pk1wM7Th1J37"
      },
      "outputs": [],
      "source": [
        "path = '/content/drive/MyDrive/EPdata'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "ztvPOBLFqpiG"
      },
      "outputs": [],
      "source": [
        "def reduce_mem(df):\n",
        "    result = df.copy()\n",
        "    for col in result.columns:\n",
        "        col_data = result[col]\n",
        "        dn = col_data.dtype.name\n",
        "        if not dn.startswith(\"datetime\"):\n",
        "            if dn == \"object\":  # only object feature has low cardinality\n",
        "                result[col] = pd.to_numeric(col_data.astype(\"category\").cat.codes, downcast=\"unsigned\")\n",
        "            elif dn.startswith(\"int\") | dn.startswith(\"uint\"):\n",
        "                if col_data.min() >= 0:\n",
        "                    result[col] = pd.to_numeric(col_data, downcast=\"unsigned\")\n",
        "                else:\n",
        "                    result[col] = pd.to_numeric(col_data, downcast='integer')\n",
        "            else:\n",
        "                result[col] = pd.to_numeric(col_data, downcast='float')\n",
        "    return result\n",
        "\n",
        "def _delete_bad_sitezero(X, y):\n",
        "    cond = (X.timestamp > '2016-05-20') | (X.site_id != 0) | (X.meter != 0)\n",
        "    X = X[cond]\n",
        "    y = y.reindex_like(X)\n",
        "    return X.reset_index(drop=True), y.reset_index(drop=True)\n",
        "\n",
        "def _extract_temporal(X, train=True):\n",
        "    X['hour'] = X.timestamp.dt.hour\n",
        "    X['weekday'] = X.timestamp.dt.weekday\n",
        "    if train:\n",
        "        # include month to create validation set, to be deleted before training\n",
        "        X['month'] = X.timestamp.dt.month \n",
        "    # month and year cause overfit, could try other (holiday, business, etc.)\n",
        "    return reduce_mem(X)\n",
        "def load_data(source='train'):\n",
        "    assert source in ['train','test']\n",
        "    df = pd.read_csv(f'{path}/{source}.csv', parse_dates=['timestamp'])\n",
        "    return reduce_mem(df)\n",
        "\n",
        "def load_building():\n",
        "    df = pd.read_csv(f'{path}/building_metadata.csv').fillna(-1)\n",
        "    return reduce_mem(df)\n",
        "\n",
        "def load_weather(source='train', fix_timezone=True, impute=True, add_lag=True):\n",
        "    assert source in ['train','test']\n",
        "    df = pd.read_csv(f'{path}/weather_{source}.csv', parse_dates=['timestamp'])\n",
        "    if fix_timezone:\n",
        "        offsets = [5,0,9,6,8,0,6,6,5,7,8,6,0,7,6,6]\n",
        "        offset_map = {site: offset for site, offset in enumerate(offsets)}\n",
        "        df.timestamp = df.timestamp - pd.to_timedelta(df.site_id.map(offset_map), unit='h')\n",
        "    if impute:\n",
        "        site_dfs = []\n",
        "        for site in df.site_id.unique():\n",
        "            if source == 'train':\n",
        "                new_idx = pd.date_range(start='2016-1-1', end='2016-12-31-23', freq='H')\n",
        "            else:\n",
        "                new_idx = pd.date_range(start='2017-1-1', end='2018-12-31-23', freq='H')\n",
        "            site_df = df[df.site_id == site].set_index('timestamp').reindex(new_idx)\n",
        "            site_df.site_id = site\n",
        "            for col in [c for c in site_df.columns if c != 'site_id']:\n",
        "                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')\n",
        "                site_df[col] = site_df[col].fillna(df[col].median())\n",
        "            site_dfs.append(site_df)\n",
        "        df = pd.concat(site_dfs)\n",
        "        df['timestamp'] = df.index\n",
        "        df = df.reset_index(drop=True)\n",
        "    if add_lag:\n",
        "        df = add_lag_features(df, window=3)\n",
        "    \n",
        "    return reduce_mem(df)\n",
        "\n",
        "def merged_dfs(source='train', fix_timezone=True, impute=True, add_lag=False):\n",
        "    df = load_data(source=source).merge(load_building(), on='building_id', how='left')\n",
        "    df = df.merge(load_weather(source=source, fix_timezone=fix_timezone, impute=impute, add_lag=add_lag),\n",
        "                 on=['site_id','timestamp'], how='left')\n",
        "    if source == 'train':\n",
        "      X = df.drop('meter_reading', axis=1)  \n",
        "      y = np.log1p(df.meter_reading)  # log-transform of target\n",
        "      return X, y\n",
        "    elif source == 'test':\n",
        "      return df\n",
        "    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n",
        "    #return X_train, X_test, y_train, y_test"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "-XoTVj9v-yup",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 444
        },
        "outputId": "05b1bc00-e793-4211-94ff-b94dea6a7d7a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "          building_id  meter           timestamp  site_id  primary_use  \\\n",
              "3340633           452      0 2016-03-03 17:00:00        3            0   \n",
              "8334597           922      0 2016-06-03 05:00:00        9            0   \n",
              "16815656          917      2 2016-11-01 06:00:00        9            0   \n",
              "16387988         1341      2 2016-10-24 14:00:00       15            0   \n",
              "3841937           741      0 2016-03-13 16:00:00        5            0   \n",
              "\n",
              "          square_feet  year_built  floor_count  air_temperature  \\\n",
              "3340633        155100      1931.0         -1.0         3.900000   \n",
              "8334597        147205        -1.0         -1.0        18.900000   \n",
              "16815656       237702        -1.0         -1.0        22.799999   \n",
              "16387988        18342      1960.0         -1.0         7.200000   \n",
              "3841937         14025      1966.0          1.0        11.000000   \n",
              "\n",
              "          cloud_coverage  dew_temperature  precip_depth_1_hr  \\\n",
              "3340633         8.000000        -6.100000                0.0   \n",
              "8334597         0.400000        17.799999                0.0   \n",
              "16815656        2.769231        21.700001                0.0   \n",
              "16387988        4.000000         3.300000                5.0   \n",
              "3841937         0.000000         2.000000                0.0   \n",
              "\n",
              "          sea_level_pressure  wind_direction  wind_speed  \n",
              "3340633          1023.099976           120.0         5.7  \n",
              "8334597          1011.799988             0.0         0.0  \n",
              "16815656         1015.066650           170.0         1.5  \n",
              "16387988         1018.461548           300.0         8.8  \n",
              "3841937          1016.400024            80.0         5.7  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-51f30a89-edbe-45e2-a86f-3eb540685135\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>building_id</th>\n",
              "      <th>meter</th>\n",
              "      <th>timestamp</th>\n",
              "      <th>site_id</th>\n",
              "      <th>primary_use</th>\n",
              "      <th>square_feet</th>\n",
              "      <th>year_built</th>\n",
              "      <th>floor_count</th>\n",
              "      <th>air_temperature</th>\n",
              "      <th>cloud_coverage</th>\n",
              "      <th>dew_temperature</th>\n",
              "      <th>precip_depth_1_hr</th>\n",
              "      <th>sea_level_pressure</th>\n",
              "      <th>wind_direction</th>\n",
              "      <th>wind_speed</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>3340633</th>\n",
              "      <td>452</td>\n",
              "      <td>0</td>\n",
              "      <td>2016-03-03 17:00:00</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>155100</td>\n",
              "      <td>1931.0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>3.900000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>-6.100000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1023.099976</td>\n",
              "      <td>120.0</td>\n",
              "      <td>5.7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8334597</th>\n",
              "      <td>922</td>\n",
              "      <td>0</td>\n",
              "      <td>2016-06-03 05:00:00</td>\n",
              "      <td>9</td>\n",
              "      <td>0</td>\n",
              "      <td>147205</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>18.900000</td>\n",
              "      <td>0.400000</td>\n",
              "      <td>17.799999</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1011.799988</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16815656</th>\n",
              "      <td>917</td>\n",
              "      <td>2</td>\n",
              "      <td>2016-11-01 06:00:00</td>\n",
              "      <td>9</td>\n",
              "      <td>0</td>\n",
              "      <td>237702</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>22.799999</td>\n",
              "      <td>2.769231</td>\n",
              "      <td>21.700001</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1015.066650</td>\n",
              "      <td>170.0</td>\n",
              "      <td>1.5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16387988</th>\n",
              "      <td>1341</td>\n",
              "      <td>2</td>\n",
              "      <td>2016-10-24 14:00:00</td>\n",
              "      <td>15</td>\n",
              "      <td>0</td>\n",
              "      <td>18342</td>\n",
              "      <td>1960.0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>7.200000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>3.300000</td>\n",
              "      <td>5.0</td>\n",
              "      <td>1018.461548</td>\n",
              "      <td>300.0</td>\n",
              "      <td>8.8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3841937</th>\n",
              "      <td>741</td>\n",
              "      <td>0</td>\n",
              "      <td>2016-03-13 16:00:00</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>14025</td>\n",
              "      <td>1966.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>11.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1016.400024</td>\n",
              "      <td>80.0</td>\n",
              "      <td>5.7</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-51f30a89-edbe-45e2-a86f-3eb540685135')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-51f30a89-edbe-45e2-a86f-3eb540685135 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-51f30a89-edbe-45e2-a86f-3eb540685135');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ],
      "source": [
        "X_train, y_train = merged_dfs(add_lag=False)\n",
        "X_train = X_train.groupby('primary_use', group_keys=False).apply(lambda X: X.sample(7500))\n",
        "X_train.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "NvrFx9jAAlNr",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "43fe7ec5-0a35-46b2-c64d-8668813e7d41"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "21"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ],
      "source": [
        "# preprocessing\n",
        "X_train, y_train = _delete_bad_sitezero(X_train, y_train)\n",
        "X_train = _extract_temporal(X_train)\n",
        "\n",
        "# remove timestamp and other unimportant features\n",
        "to_drop = ['timestamp','sea_level_pressure','wind_direction','wind_speed', 'precip_depth_1_hr', 'year_built', 'square_feet']\n",
        "X_train.drop(to_drop, axis=1, inplace=True)\n",
        "\n",
        "gc.collect()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "7Dmxi2Ni1tO-",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 488
        },
        "outputId": "e11f696a-9178-4386-97b1-568d369d3d12"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        building_id  meter  site_id  primary_use  floor_count  \\\n",
              "0               452      0        3            0         -1.0   \n",
              "1               922      0        9            0         -1.0   \n",
              "2               917      2        9            0         -1.0   \n",
              "3              1341      2       15            0         -1.0   \n",
              "4               741      0        5            0          1.0   \n",
              "...             ...    ...      ...          ...          ...   \n",
              "116692          413      0        3           15         -1.0   \n",
              "116693          829      0        8           15          1.0   \n",
              "116694          505      0        3           15         -1.0   \n",
              "116695          505      0        3           15         -1.0   \n",
              "116696          164      0        2           15         -1.0   \n",
              "\n",
              "        air_temperature  cloud_coverage  dew_temperature  hour  weekday  month  \n",
              "0              3.900000        8.000000        -6.100000    17        3      3  \n",
              "1             18.900000        0.400000        17.799999     5        4      6  \n",
              "2             22.799999        2.769231        21.700001     6        1     11  \n",
              "3              7.200000        4.000000         3.300000    14        0     10  \n",
              "4             11.000000        0.000000         2.000000    16        6      3  \n",
              "...                 ...             ...              ...   ...      ...    ...  \n",
              "116692         4.400000        8.000000        -1.100000     5        2      1  \n",
              "116693        33.299999        4.000000        22.200001    12        4      6  \n",
              "116694        -4.400000        8.000000        -6.100000    14        4      1  \n",
              "116695         3.900000        2.000000        -7.200000    18        6     11  \n",
              "116696        37.200001        0.000000         0.600000    17        0      9  \n",
              "\n",
              "[116697 rows x 11 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-ecb35fde-3729-400a-a5f0-374fbfbc9a84\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>building_id</th>\n",
              "      <th>meter</th>\n",
              "      <th>site_id</th>\n",
              "      <th>primary_use</th>\n",
              "      <th>floor_count</th>\n",
              "      <th>air_temperature</th>\n",
              "      <th>cloud_coverage</th>\n",
              "      <th>dew_temperature</th>\n",
              "      <th>hour</th>\n",
              "      <th>weekday</th>\n",
              "      <th>month</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>452</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>3.900000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>-6.100000</td>\n",
              "      <td>17</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>922</td>\n",
              "      <td>0</td>\n",
              "      <td>9</td>\n",
              "      <td>0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>18.900000</td>\n",
              "      <td>0.400000</td>\n",
              "      <td>17.799999</td>\n",
              "      <td>5</td>\n",
              "      <td>4</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>917</td>\n",
              "      <td>2</td>\n",
              "      <td>9</td>\n",
              "      <td>0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>22.799999</td>\n",
              "      <td>2.769231</td>\n",
              "      <td>21.700001</td>\n",
              "      <td>6</td>\n",
              "      <td>1</td>\n",
              "      <td>11</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1341</td>\n",
              "      <td>2</td>\n",
              "      <td>15</td>\n",
              "      <td>0</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>7.200000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>3.300000</td>\n",
              "      <td>14</td>\n",
              "      <td>0</td>\n",
              "      <td>10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>741</td>\n",
              "      <td>0</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>11.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>16</td>\n",
              "      <td>6</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>116692</th>\n",
              "      <td>413</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>15</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>4.400000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>-1.100000</td>\n",
              "      <td>5</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>116693</th>\n",
              "      <td>829</td>\n",
              "      <td>0</td>\n",
              "      <td>8</td>\n",
              "      <td>15</td>\n",
              "      <td>1.0</td>\n",
              "      <td>33.299999</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>22.200001</td>\n",
              "      <td>12</td>\n",
              "      <td>4</td>\n",
              "      <td>6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>116694</th>\n",
              "      <td>505</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>15</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>-4.400000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>-6.100000</td>\n",
              "      <td>14</td>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>116695</th>\n",
              "      <td>505</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>15</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>3.900000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>-7.200000</td>\n",
              "      <td>18</td>\n",
              "      <td>6</td>\n",
              "      <td>11</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>116696</th>\n",
              "      <td>164</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>15</td>\n",
              "      <td>-1.0</td>\n",
              "      <td>37.200001</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.600000</td>\n",
              "      <td>17</td>\n",
              "      <td>0</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>116697 rows × 11 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ecb35fde-3729-400a-a5f0-374fbfbc9a84')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-ecb35fde-3729-400a-a5f0-374fbfbc9a84 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-ecb35fde-3729-400a-a5f0-374fbfbc9a84');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ],
      "source": [
        "X_train"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "ZBOtJzRVB2Oz",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fe0e9b5f-fdcb-4214-ba56-f527bcee56e4"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['building_id',\n",
              " 'meter',\n",
              " 'site_id',\n",
              " 'primary_use',\n",
              " 'floor_count',\n",
              " 'air_temperature',\n",
              " 'cloud_coverage',\n",
              " 'dew_temperature',\n",
              " 'hour',\n",
              " 'weekday',\n",
              " 'month']"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ],
      "source": [
        "list(X_train.columns)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "FRKueZmSV0FK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bec8f6a5-dec6-4306-eb68-5fc8d8a9cb4e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 116697 entries, 0 to 116696\n",
            "Data columns (total 12 columns):\n",
            " #   Column           Non-Null Count   Dtype  \n",
            "---  ------           --------------   -----  \n",
            " 0   building_id      116697 non-null  uint16 \n",
            " 1   meter            116697 non-null  uint8  \n",
            " 2   site_id          116697 non-null  uint8  \n",
            " 3   primary_use      116697 non-null  uint8  \n",
            " 4   floor_count      116697 non-null  float32\n",
            " 5   air_temperature  116697 non-null  float32\n",
            " 6   cloud_coverage   116697 non-null  float32\n",
            " 7   dew_temperature  116697 non-null  float32\n",
            " 8   hour             116697 non-null  uint8  \n",
            " 9   weekday          116697 non-null  uint8  \n",
            " 10  month            116697 non-null  uint8  \n",
            " 11  meter_reading    116697 non-null  float32\n",
            "dtypes: float32(5), uint16(1), uint8(6)\n",
            "memory usage: 3.1 MB\n"
          ]
        }
      ],
      "source": [
        "X = pd.concat([X_train, y_train], axis=1)\n",
        "\n",
        "del X_train, y_train\n",
        "gc.collect()\n",
        "\n",
        "X.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "FZkmozgSAwot"
      },
      "outputs": [],
      "source": [
        "Y = X['meter_reading']"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dropping = ['meter_reading']\n",
        "X.drop(dropping, axis=1, inplace=True)"
      ],
      "metadata": {
        "id": "XYznuLmyZkB1"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "riMZ89_KZ6b2",
        "outputId": "de21ae40-d1c3-4a59-aaa5-5b4f0fa388df"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(116697, 11)"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "yq85E7s6KSVB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "92328ee9-e6de-4fc4-e66d-510e0b7a74e3"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    4.981344\n",
              "1    4.442651\n",
              "2    4.423048\n",
              "3    7.519953\n",
              "4    1.098612\n",
              "Name: meter_reading, dtype: float32"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ],
      "source": [
        "Y.head()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)"
      ],
      "metadata": {
        "id": "3uqND60H_HNh"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "8T7g46E44dHT",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b754f834-d9de-46d7-a208-9ea3ebd76b63"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(93357, 11)"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ],
      "source": [
        "X_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "pamv5tum2ahX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "00d94d1a-e063-413a-834a-60920912cd52"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(23340, 11)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ],
      "source": [
        "X_test.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "tGQRtJTU2d7D",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "47142167-cf67-4822-dc6a-78e68d7aed42"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(93357,)"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ],
      "source": [
        "y_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "N_UXOMI_2hZV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1130afb7-4cb4-45b0-ae65-3fdd48ca417f"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(23340,)"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ],
      "source": [
        "y_test.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "4HwWNuc12k4S"
      },
      "outputs": [],
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "scaler = StandardScaler()\n",
        "scaler.fit(X_train)\n",
        "x_train = scaler.transform(X_train)\n",
        "x_test = scaler.transform(X_test)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Decission Tree**"
      ],
      "metadata": {
        "id": "zad3dEMK-Woc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "parameters = {'max_depth': [3,5,7,9,11,15]}\n",
        "\n",
        "\n",
        "decission_tree = GridSearchCV(estimator = DecisionTreeRegressor(),\n",
        "                        param_grid = parameters,\n",
        "                        cv = 3, \n",
        "                        scoring = 'neg_mean_squared_error',\n",
        "                        verbose = 1,\n",
        "                        return_train_score = True,\n",
        "                        n_jobs = -1)\n",
        "\n",
        "decission_tree.fit(x_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XMUIFpSa-av_",
        "outputId": "a83c87c4-42ed-4cde-c8ba-60d6d40e3e8e"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Fitting 3 folds for each of 6 candidates, totalling 18 fits\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GridSearchCV(cv=3, estimator=DecisionTreeRegressor(), n_jobs=-1,\n",
              "             param_grid={'max_depth': [3, 5, 7, 9, 11, 15]},\n",
              "             return_train_score=True, scoring='neg_mean_squared_error',\n",
              "             verbose=1)"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Best Decision Tree Estimator \",decission_tree.best_estimator_)\n",
        "print(\"Best Decision Tree Paramteres are : \", decission_tree.best_params_)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XZzrq0rb-cAc",
        "outputId": "061f0710-6182-4429-f8e7-4eacbab2781d"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Best Decision Tree Estimator  DecisionTreeRegressor(max_depth=15)\n",
            "Best Decision Tree Paramteres are :  {'max_depth': 15}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "DecissionTree = DecisionTreeRegressor(max_depth=15)\n",
        "DecissionTree.fit(x_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WtYJRT8y-gBf",
        "outputId": "5e70152a-63a0-4d65-de10-005a53b11fef"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DecisionTreeRegressor(max_depth=15)"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Vb_GVz5y8t98"
      },
      "source": [
        "**SVM**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "NNXCuf1_5QFl",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "870a9cc0-63fa-459d-c235-dec625b005f7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mean_squared_error:  1.9704024395595257\n",
            "mean_absolute_error:  1.2203734290216264\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/svm/_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.\n",
            "  warnings.warn(\n"
          ]
        }
      ],
      "source": [
        "from sklearn.metrics import mean_squared_error,mean_absolute_error\n",
        "from sklearn.svm import LinearSVR\n",
        "lsvr = LinearSVR(random_state=0, tol=1e-5)\n",
        "lsvr.fit(x_train,y_train)\n",
        "pred=lsvr.predict(x_test)  \n",
        "print(\"mean_squared_error: \",np.sqrt(mean_squared_error(y_test, pred))) \n",
        "print(\"mean_absolute_error: \", np.sqrt(mean_absolute_error(y_test, pred)))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "xiS8aS8D8w53",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0d596452-4d04-4e2f-fa62-17619eba5998"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mean_squared_error:  1.6392418827550848\n",
            "mean_absolute_error:  1.0753882232443617\n"
          ]
        }
      ],
      "source": [
        "from sklearn import svm\n",
        "\n",
        "svr_model = svm.SVR(gamma = 'scale')\n",
        "fit_model = svr_model.fit(x_train , y_train)\n",
        "pred=svr_model.predict(x_test)  \n",
        "print(\"mean_squared_error: \",np.sqrt(mean_squared_error(y_test, pred))) \n",
        "print(\"mean_absolute_error: \", np.sqrt(mean_absolute_error(y_test, pred))) "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NsnGH-DJ9rpy"
      },
      "source": [
        "**Ensemble Learner**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "Y4uggorK9u1x"
      },
      "outputs": [],
      "source": [
        "estimators = [\n",
        "              ('lsvr', LinearSVR()),\n",
        "              ('svr_model', SVR()),\n",
        "              ('decission_tree', DecisionTreeRegressor())\n",
        "              ]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "metadata": {
        "id": "m9BMOl5e9vpU"
      },
      "outputs": [],
      "source": [
        "final_estimator = GradientBoostingRegressor(\n",
        "        n_estimators=150, subsample=0.5, min_samples_leaf=25, max_features=1,\n",
        "        random_state=42)\n",
        "Ensemble_model = StackingRegressor(\n",
        "      estimators=estimators,\n",
        "      final_estimator=final_estimator)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "id": "3Kck7n8_9zcN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b6df70ae-dd2b-41a9-90a2-7b2168126ad7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mean_squared_error:  1.025914808759214\n",
            "mean_absolute_error:  0.7497573217291433\n"
          ]
        }
      ],
      "source": [
        "Ensemble_model.fit(x_train, y_train)\n",
        "pred=Ensemble_model.predict(x_test)  \n",
        "print(\"mean_squared_error: \",np.sqrt(mean_squared_error(y_test, pred))) \n",
        "print(\"mean_absolute_error: \", np.sqrt(mean_absolute_error(y_test, pred)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TBS7VzZiDaR-"
      },
      "source": [
        "**Model Evaluation and Selection**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "ORAU7Q9JDZV4"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import TimeSeriesSplit\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "from sklearn.model_selection import cross_val_score, cross_val_predict\n",
        "from sklearn import metrics"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "metadata": {
        "id": "tBrmNOB_DhoU"
      },
      "outputs": [],
      "source": [
        "def evaluate(model, test_features, test_labels):\n",
        "    predictions = model.predict(test_features)\n",
        "    errors = abs(predictions - test_labels)\n",
        "    adds = abs(predictions + test_labels)/2\n",
        "    smape = np.mean(errors / adds) * 100\n",
        "    mape = 100 * np.mean(errors / test_labels)\n",
        "    r_score = 100*r2_score(test_labels,predictions)\n",
        "    accuracy = 100 - smape\n",
        "    print(model,'\\n')\n",
        "    print('Average Error(Mean Absolute Error)       : {:0.4f} degrees'.format(np.mean(errors)))\n",
        "    print('Variance score R^2  : {:0.2f}%' .format(r_score))\n",
        "    print('Accuracy            : {:0.2f}%\\n'.format(accuracy)) "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "id": "axDmOYWvDlcS",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2bb3aa57-cc9a-4e98-eddf-66764ed56168"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "StackingRegressor(estimators=[('lsvr', LinearSVR()), ('svr_model', SVR()),\n",
            "                              ('decission_tree', DecisionTreeRegressor())],\n",
            "                  final_estimator=GradientBoostingRegressor(max_features=1,\n",
            "                                                            min_samples_leaf=25,\n",
            "                                                            n_estimators=150,\n",
            "                                                            random_state=42,\n",
            "                                                            subsample=0.5)) \n",
            "\n",
            "Average Error(Mean Absolute Error)       : 0.5621 degrees\n",
            "Variance score R^2  : 73.42%\n",
            "Accuracy            : 76.24%\n",
            "\n"
          ]
        }
      ],
      "source": [
        "evaluate(Ensemble_model, x_test, y_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "id": "2cFEy28LDtJL",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6ebf1160-add2-44e2-c6dd-34d1e265294e"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "23340"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ],
      "source": [
        "y1_pred = Ensemble_model.predict(x_test)\n",
        "len(y1_pred)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "metadata": {
        "id": "YY8OSEpiDt0x",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5cabdf52-3e00-42ef-c478-10307581fb3e"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([2.59257937, 7.12522627, 5.10585489, ..., 4.32751278, 2.94264055,\n",
              "       2.53535304])"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ],
      "source": [
        "y1_pred"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "id": "2jCsOW-8Dxlu",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 419
        },
        "outputId": "31ed883a-972f-4413-ecc9-79ba0a2b9261"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7f4137df3b80>"
            ]
          },
          "metadata": {},
          "execution_count": 33
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1440x576 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABHcAAAHSCAYAAABmRifhAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeXxcdb3/8dfJTJLJZJusbZaWtlC27qWlQAXKDoKIQKnSy+JVURDUixso/qhXQa6AC15ARBC9VLYqIMpSoUVkLS1LoTuFLmm2SZrMlm0yc35/nE6WZl/nTPJ+Ph59TGfJ6Tdpkjnzns/n8zVM00RERERERERERBJTUrwXICIiIiIiIiIig6dwR0REREREREQkgSncERERERERERFJYAp3REREREREREQSmMIdEREREREREZEEpnBHRERERERERCSBOUfioPn5+eaUKVNG4tAiIiIiIiIiIuPShg0bakzTLDj49hEJd6ZMmcL69etH4tAiIiIiIiIiIuOSYRi7u7tdbVkiIiIiIiIiIglM4Y6IiIiIiIiISAJTuCMiIiIiIiIiksBGZOaOiIiIiIiIiAxOOBymrKyMpqameC9F4sTlclFaWkpycnK/Hq9wR0RERERERMRGysrKyMzMZMqUKRiGEe/lyCgzTZPa2lrKysqYOnVqvz5GbVkiIiIiIiIiNtLU1EReXp6CnXHKMAzy8vIGVLmlcEdERERERETEZhTsjG8D/f9XuCMiIiIiIiIibWpra5k7dy5z585l4sSJlJSUtF1vaWkZ1n+rvr6ee+65Z9iOl5GRMWzHSiSauSMiIiIiIiIibfLy8njvvfcAWLFiBRkZGXznO9/p8+NaW1txOgcWM8TCnWuuuWZQaxWLKndEREREREREpFf3338/CxcuZM6cOVx00UU0NDQAcOWVV/K1r32NRYsW8b3vfY+dO3dy3HHHMWvWLG666aZOlTS33347CxcuZPbs2dx8880A3HDDDezcuZO5c+fy3e9+t9O/ecMNN3D33Xe3XV+xYgV33HEHwWCQ0047jfnz5zNr1iyefvrpLut9+eWXOe+889quX3vttTz00EMAbNiwgZNPPpljjjmGs846i4qKimH7OsWLKndEREREREREbOpb34IDRTTDZu5c+NWvBvYxF154IV/5ylcAuOmmm3jggQe47rrrAGt3r9dffx2Hw8F5553HN7/5Tb7whS/w29/+tu3jV69ezY4dO1i3bh2maXL++efzyiuvcNttt/Hhhx+2VQp1tGzZMr71rW/x9a9/HYDHH3+cF154AZfLxZNPPklWVhY1NTUcd9xxnH/++f2aUxMOh7nuuut4+umnKSgo4LHHHuOHP/whDz744MC+IDajcEdEREREREREevXhhx9y0003UV9fTzAY5Kyzzmq7b+nSpTgcDgDeeOMNnnrqKQAuvfTStnau1atXs3r1aubNmwdAMBhkx44dTJ48ucd/c968eVRXV1NeXo7X6yUnJ4dJkyYRDof5wQ9+wCuvvEJSUhL79u2jqqqKiRMn9vl5bNu2jQ8//JAzzjgDgEgkQlFR0eC+KDaicEdERERERETEpgZaYTNSrrzySp566inmzJnDQw89xMsvv9x2X3p6ep8fb5omN954I1/96lc73b5r165eP27p0qWsWrWKyspKli1bBsDKlSvxer1s2LCB5ORkpkyZ0mXbcKfTSTQabbseu980TWbMmMEbb7zR55oTiWbuiIiIiIiIiEivAoEARUVFhMNhVq5c2ePjjjvuOP7yl78A8Oijj7bdftZZZ/Hggw8SDAYB2LdvH9XV1WRmZhIIBHo83rJly3j00UdZtWoVS5cuBcDn81FYWEhycjJr165l9+7dXT7ukEMOYfPmzTQ3N1NfX89LL70EwBFHHIHX620Ld8LhMJs2bRrgV8N+FO6IiIiIiIiISK9+8pOfsGjRIhYvXsyRRx7Z4+N+9atf8Ytf/ILZs2fz0UcfkZ2dDcCZZ57JpZdeyvHHH8+sWbO4+OKLCQQC5OXlsXjxYmbOnNlloDLAjBkzCAQClJSUtLVPLV++nPXr1zNr1iz+9Kc/dbueSZMmcckllzBz5kwuueSStnawlJQUVq1axfe//33mzJnD3Llzef3114fjSxRXhmmaw37QBQsWmOvXrx/244qIiIiIiIiMdVu2bOGoo46K9zIGpaGhgbS0NAzD4NFHH+WRRx7pdjcr6Vt33weGYWwwTXPBwY/VzB0RERERERERGRYbNmzg2muvxTRNPB5Pwu9ClSgU7oiIiIiIJLjzz4cZM+BnP4v3SkRkvDvxxBN5//33472McUfhjoiIiIhIgtu4EQwj3qsQEZF40UBlEREREZEEFwpBY2O8VyEiIvGicEdEREREJMEFg9DUFO9ViIhIvCjcERERERFJYJGIFeyockdEZPxSuCMiIiIiksAaGqxLVe6IyHByOBzMnTu37c9tt9026mtYsWIFd9xxR5fbd+3axcyZMwd0rIyMjOFali1poLKIiIiISAILhaxLVe6IyHBKS0vjvffei/cybKu1tRWnc2iRSiQSweFwDMt6VLkjIiIiIpLAFO6IyGiaMmUKN998M/Pnz2fWrFls3boVgH/9619tVT7z5s0jEAgAcPvtt7Nw4UJmz57NzTffDFiVN0ceeSRXXnklhx9+OMuXL+fFF19k8eLFTJ8+nXXr1rX9e++//z7HH38806dP5/777++ynkgkwne/+922f+O+++7r9+eyc+dOzj77bI455hhOPPHEts/lmWeeYdGiRcybN4/TTz+dqqoqwKokuuyyy1i8eDGXXXYZK1as4D//8z9ZsmQJ06ZN46677mo79sMPP8yxxx7L3Llz+epXv0okEgGsCqJvf/vbzJkzhzfeeGMgX/peqXJHRERERCSBxcIdtWWJjF1LHlrS5bZLZlzCNQuvoSHcwKdXfrrL/VfOvZIr515JTUMNFz9+caf7Xr7y5T7/zcbGRubOndt2/cYbb2TZsmUA5Ofn884773DPPfdwxx138Pvf/5477riDu+++m8WLFxMMBnG5XKxevZodO3awbt06TNPk/PPP55VXXmHy5Ml89NFHPPHEEzz44IMsXLiQP//5z7z66qv87W9/49Zbb+Wpp54CYOPGjbz55puEQiHmzZvHueee22mdDzzwANnZ2bz99ts0NzezePFizjzzTKZOndrn53jVVVfx29/+lunTp/PWW29xzTXXsGbNGj71qU/x5ptvYhgGv//97/n5z3/OnXfeCcDmzZt59dVXSUtLY8WKFWzdupW1a9cSCAQ44ogjuPrqq/noo4947LHHeO2110hOTuaaa65h5cqVXH755YRCIRYtWtR2vOGicEdEREREJIGpckdERkJvbVkXXnghAMcccwx//etfAVi8eDHXX389y5cv58ILL6S0tJTVq1ezevVq5s2bB0AwGGTHjh1MnjyZqVOnMmvWLABmzJjBaaedhmEYzJo1i127drX9W5/97GdJS0sjLS2NU045hXXr1nUKnVavXs3GjRtZtWoVAD6fjx07dvQZ7gSDQV5//XWWLl3adltzczMAZWVlLFu2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knjGFO4YYy40xuwwxuw2xtzs9aBERLyQkQFf+ALs3g3f/CY8/DAsWAA33QS1tX37NTdDd3dqVe6ADXY0JUtEREREJPGMGu4YY3zAT4G1wCLgKmPMIq8HJiLilYIC+MEPYNcuuOYauP12O2Xrxz+Gtra+FbVSLdzx+xXuiIiIiIgkorFU7vwVsNtxnD2O43QAvwUu9XZYIiLeKy2FX/7STtP68Ifh61+H+fPh7rvt66kY7mhaloiIiIhI4hlLuFMKHOz3+P2e50REksKpp8Jjj8HGjRAKwfe+Z59PxXBHlTsiIiIiIoknfaLeyBhzPXA9wMyZMyfqbUVEoubcc+Gll+DBB+3XWWfFekTRde21kD5hZwUREREREYkW4zjOyDsYczbwHcdxLuh5/E8AjuP8cLifWbZsmbN169aJHKeIiIiIiIiISEozxrzqOM6y458fy7SsV4B5xphyY0wmcCXw6EQPUERERERERERExm/UAnzHcbqMMf8IPAn4gHscx9nu+chERERERERERGRUY+qu4DjO48DjHo9FRERERERERETGaSzTskREREREREREJE4p3BERERERERERSWAKd0REREREREREEpjCHRERERERERGRBKZwR0REREREREQkgSncERERERERERFJYAp3REREREREREQSmMIdEREREREREZEEpnBHRERERERERCSBKdwREREREREREUlgCndERERERERERBKYwh0RERERERERkQSmcEdEREREREREJIEp3BERERERERERSWAKd0REREREREREEphxHGfi39SYamD/hL9x9E0CjsV6ECIJRJ8ZkfHRZ0ZkfPSZERkffWZExicRPjOzHMeZfPyTnoQ7ycIYs9VxnGWxHodIotBnRmR89JkRGR99ZkTGR58ZkfFJ5M+MpmWJiIiIiIiIiCQwhTsiIiIiIiIiIglM4c7I7o71AEQSjD4zIuOjz4zI+OgzIzI++syIjE/CfmbUc0dEREREREREJIGpckdEREREREREJIEp3BmGMeZCY8wOY8xuY8zNsR6PSLwxxswwxmwyxrxtjNlujPlSz/PFxpinjTG7erZFsR6rSLwwxviMMduMMY/1PC43xrzUc675f8aYzFiPUSSeGGMKjTEPGWPeNca8Y4w5W+cZkeEZY77cc132ljHmfmNMts41In2MMfcYY44aY97q99yQ5xVj3dHz2XnDGLMkdiMfncKdIRhjfMBPgbXAIuAqY8yi2I5KJO50ATc5jrMIWA7c0PM5uRnY4DjOPGBDz2MRsb4EvNPv8b8C/9dxnLlAHfDZmIxKJH79BHjCcZyFwAewnx+dZ0SGYIwpBb4ILHMc51TAB1yJzjUi/f0ncOFxzw13XlkLzOv5uh74eZTGeEIU7gztr4DdjuPscRynA/gtcGmMxyQSVxzHOeI4TkXP943YC+5S7Gfl1z27/Rq4LDYjFIkvxpjpwEXAL3oeG2A18FDPLvq8iPRjjCkAVgG/BHAcp8NxnHp0nhEZSTqQY4xJB3KBI+hcI9LLcZzngNrjnh7uvHIp8F+O9SJQaIyZGp2Rjp/CnaGVAgf7PX6/5zkRGYIxpgw4E3gJCDmOc6TnpUogFKNhicSb24GvA5GexyVAveM4XT2Pda4RGagcqAZ+1TOd8RfGmDx0nhEZkuM4h4D/DRzAhjph4FV0rhEZzXDnlYTKBRTuiMhJMcb4gf8GbnQcp6H/a45djk9L8knKM8ZcDBx1HOfVWI9FJIGkA0uAnzuOcybQzHFTsHSeEenT0yfkUmwwOg3IY/D0ExEZQSKfVxTuDO0QMKPf4+k9z4lIP8aYDGyws95xnN/1PF3lliv2bI/GanwiceTDwCXGmH3Yqb6rsb1ECntK50HnGpHjvQ+87zjOSz2PH8KGPTrPiAztPGCv4zjVjuN0Ar/Dnn90rhEZ2XDnlYTKBRTuDO0VYF5PZ/lMbCOyR2M8JpG40tMv5JfAO47j3NbvpUeBv+35/m+B30d7bCLxxnGcf3IcZ7rjOGXYc8pGx3GuBjYBn+zZTZ8XkX4cx6kEDhpjFvQ8tQZ4G51nRIZzAFhujMntuU5zPzM614iMbLjzyqPA3/SsmrUcCPebvhV3jK06kuMZY9Zh+yP4gHscx/lBjIckEleMMSuA54E36esh8k1s350HgJnAfuAKx3GOb1omkrKMMecAX3Uc52JjzGxsJU8xsA24xnGc9liOTySeGGPOwDYhzwT2AP8Te3NS5xmRIRhjvgt8Cruq6Tbgc9geITrXiADGmPuBc4BJQBXwv4BHGOK80hOS3oWd3tgC/E/HcbbGYtxjoXBHRERERERERCSBaVqWiIiIiIiIiEgCU7gjIiIiIiIiIpLAFO6IiIiIiIiIiCQwhTsiIiIiIiIiIglM4Y6IiIiIiIiISAJTuCMiIiIiIiIiksAU7oiIiIiIiIiIJDCFOyIiIiIiIiIiCez/A+8NHABt0IEyAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "fig = plt.figure(figsize=(20,8))\n",
        "plt.plot(y_test[:100].values,label='Target value',color='b')\n",
        "plt.plot(y1_pred[:100],label='Ensemble Learner ', linestyle='--', color='g')\n",
        "\n",
        "plt.legend(loc=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "metadata": {
        "id": "XxXruHsLD3xZ"
      },
      "outputs": [],
      "source": [
        "# save the file for deployment\n",
        "\n",
        "import pickle\n",
        "file = open('EnsembleModel.pkl','wb')\n",
        "pickle.dump(Ensemble_model,file)\n",
        "file.close()"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "machine_shape": "hm",
      "provenance": [],
      "include_colab_link": true
    },
    "gpuClass": "standard",
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}