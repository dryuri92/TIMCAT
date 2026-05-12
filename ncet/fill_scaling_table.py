"""
ncet/fill_scaling_table.py  —  hardened rewrite
================================================
Changes vs original:
  1. Replaced all deprecated DataFrame.append() with pd.concat().
  2. NaN-indexed rows are dropped from every DataFrame right after loading
     so that str.match() / isin() / == never produce NA-valued boolean masks
     (which cause ValueError: "Cannot mask with non-boolean array containing
     NA / NaN values" on .loc[] assignment).
  3. _is_nan()     — universal NaN test covering None, float NaN, numpy scalar
                     NaN, and 0-d numpy arrays; safe on any Python type.
  4. _clean_mask() — converts any mask (Series, ndarray, scalar) to a pure
                     bool Series with NaN → False before it touches .loc[].
  5. _safe_loc_set() runs _clean_mask() internally as a last-resort guard.
  6. "Method" columns are filled with "" (not NaN) after loading so that
     == comparisons always yield a clean bool Series.
  7. All .loc[] / .at[] index accesses are existence-checked first.
  8. plant_characteristics key accesses use _get_pc() with safe defaults.
  9. Division-by-zero guarded via _safe_divide() and .replace(0, np.nan).
 10. f-strings replace .format() for readability; no logic changes.
"""

import pandas as pd
import numpy as np
from os.path import join as pjoin

from .bldg_features import eval_bldg
from .special_cases import cost_multipliers
from .material_use_uncertainty import material_use_uncertainty


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_nan(value) -> bool:
    """
    Universal missing-value test.
    Returns True for: Python None, float NaN, numpy scalar NaN (any dtype),
    and 0-d numpy arrays whose single element is NaN.
    Returns False for strings, lists, dicts, non-NaN numbers, etc.
    """
    if value is None:
        return True
    try:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return bool(np.isnan(value))
            return False      # non-scalar arrays are never "a NaN value"
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _clean_mask(mask, index: pd.Index) -> pd.Series:
    """
    Convert *mask* to a pure bool Series aligned to *index*.
    NaN entries (produced by str.match / == on NaN-containing indices)
    become False so those rows are never selected.
    Called before every .loc[] write.
    """
    if isinstance(mask, pd.Series):
        return mask.fillna(False).astype(bool)
    if isinstance(mask, np.ndarray):
        s = pd.Series(mask, index=index)
        return s.fillna(False).astype(bool)
    return pd.Series(bool(mask), index=index)


def _drop_nan_index_rows(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Remove rows whose index is NaN.
    pandas reads blank Excel / CSV rows as NaN-indexed rows.  Keeping them
    causes str.match() to return NaN instead of True/False, which then raises
    ValueError: "Cannot mask with non-boolean array containing NA / NaN values".
    """
    nan_mask = pd.isna(df.index)
    n = int(nan_mask.sum())
    if n:
        print(f"  [INFO] Dropping {n} NaN-indexed row(s) from {label!r}")
        df = df.loc[~nan_mask]
    return df

def _fill_nan_index_rows(df: pd.DataFrame, label: str = "",
                         placeholder: str = "__nan_row__") -> pd.DataFrame:
    """
    Replace NaN index values with a harmless placeholder string instead of
    dropping the row.  This preserves all data (including numeric columns
    like cost values that would otherwise be lost) while ensuring that
    str.match() always returns a clean bool — never NaN — because the index
    no longer contains any missing values.
 
    The placeholder begins with "__" so it will never accidentally match a
    real EEDB account code (which always start with "A.").
 
    Numeric columns that are NaN in a blank row are left as-is; callers
    that need a numeric default (e.g. 0.0) should apply .fillna() themselves
    on the specific column they care about.
    """
    nan_mask = pd.isna(df.index)
    n = int(nan_mask.sum())
    if n:
        print(f"  [INFO] Filling {n} NaN index value(s) with "
              f"{placeholder!r} in {label!r} — rows are kept, not dropped")
        new_index = df.index.astype(object).where(~nan_mask, other=placeholder)
        df = df.copy()
        df.index = new_index
    return df


def _safe_loc_set(df: pd.DataFrame, mask, col: str, value,
                  label: str = "") -> None:
    """
    df.loc[mask, col] = value with three protection layers:
      1. _clean_mask() ensures the mask is a pure bool Series (NaN -> False).
      2. Skip entirely if nothing is selected.
      3. Catch and log any remaining exception.
    """
    clean = _clean_mask(mask, df.index)
    if not clean.any():
        return
    try:
        df.loc[clean, col] = value
    except Exception as exc:
        print(f"  [WARN] _safe_loc_set failed for {label!r}, col={col!r}: {exc}")


def _get_pc(plant_characteristics: dict, key: str,
            default=None, label: str = ""):
    """Fetch from plant_characteristics with a warning when key is absent."""
    if key not in plant_characteristics:
        ctx = f" (context: {label})" if label else ""
        print(f"  [WARN] plant_characteristics missing key {key!r}{ctx}")
        return default
    return plant_characteristics[key]


def _safe_divide(numerator, denominator, label: str = "") -> float:
    """numerator / denominator; returns 1.0 with a warning on 0 or NaN denom."""
    try:
        if _is_nan(denominator) or denominator == 0:
            print(f"  [WARN] Division by zero/NaN for {label!r}; returning 1.0")
            return 1.0
        return numerator / denominator
    except Exception as exc:
        print(f"  [WARN] _safe_divide error for {label!r}: {exc}; returning 1.0")
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Main function
# ─────────────────────────────────────────────────────────────────────────────

def fill_scaling_table(path, fname, base, scalars_dict, scaling_table=None):

    # ── Load or prepare the scaling table ────────────────────────────────────
    if scaling_table is None:
        scaling_table = pd.read_csv(
            pjoin(path, "input_scaling_exponents.csv"),
            header=0,
            index_col="Account",
        )
    else:
        #scaling_table["Account"] = pd.Series(dtype=object)
        scaling_table.set_index("Account", inplace=True)

    # Drop NaN-indexed rows immediately — root cause of the ValueError.
    #scaling_table = _fill_nan_index_rows(scaling_table, "scaling_table")

    scaling_table["Option"]                      = 1
    scaling_table["New Base Unit Value"]         = 0.0
    scaling_table["Multipliers"]                 = 1.0
    scaling_table["Factory Equipment Cost Mult"] = 1.0
    scaling_table["Site Labor Hours Mult"]       = 1.0
    scaling_table["Site Labor Cost Mult"]        = 1.0
    scaling_table["Site Material Cost Mult"]     = 1.0
    scaling_table["Count per plant"]             = 1
    scaling_table["New Cost"]                    = 0.0
    inside_dict: dict = {}
    # ── Plant characteristics ─────────────────────────────────────────────────
    plant_characteristics: dict = pd.read_excel(
        pjoin(path, fname),
        sheet_name="PlantCharacteristics",
        header=None,
        skiprows=[0],
        index_col=0,
    ).to_dict()[1]

    plant_characteristics.setdefault("SPC One sided",    [])
    plant_characteristics.setdefault("SPC Two sided",    [])
    plant_characteristics.setdefault("SPC Area",         [])
    plant_characteristics.setdefault("Grade 80",         [])
    plant_characteristics.setdefault("Grade 100",        [])
    plant_characteristics.setdefault("Containment type", ["Steel lined concrete"])
    plant_characteristics.setdefault("sc1_BV",           0)
    plant_characteristics.setdefault("sc1_concrete",     0)

    concrete = 0
    bv_accounts_225 = ["A.212.", "A.213.", "A.215.", "A.216.", "A.217."]
    bv_225 = 0

    # ── Load sheets, dropping NaN-index rows and sanitizing Method column ─────
    _skw = dict(header=0, skiprows=[0], index_col="Account")

    def _load_sheet(sheet_name: str) -> pd.DataFrame:
        df = pd.read_excel(pjoin(path, fname), sheet_name=sheet_name, **_skw)
        #df = _fill_nan_index_rows(df, sheet_name)
        # Make Method a plain string — NaN cells become "" so that
        # df["Method"] == "anything" always returns a clean bool Series.
        if "Method" in df.columns:
            df["Method"] = df["Method"].fillna("").astype(str)
        return df

    df21 = _load_sheet("21-Structures&Improvements")
    df22 = _load_sheet("22-ReactorEquipment")
    df23 = _load_sheet("23-TurbineEquipment")
    df24 = _load_sheet("24-ElectricalEquipment")
    df25 = _load_sheet("25-MiscEquipment")
    df26 = _load_sheet("26-HeatRejectionSystem")

    plant_characteristics["New Bldg"] = df21["SSCs moved to"]

    if "Rebar density" in df21.columns and any(df21["Rebar density"] != "Default"):
        plant_characteristics["Rebar table"] = (
            df21.loc[df21["Rebar density"] != "Default", "Rebar density"].to_dict()
        )

    # ── Pre-computed option-type masks ────────────────────────────────────────
    # The "Option 1" column may contain NaN for blank rows; clean each mask.
    def _opt1_mask(value: str) -> pd.Series:
        return _clean_mask(scaling_table["Option 1"] == value, scaling_table.index)

    ibv  = _opt1_mask("Building volume")
    isba = _opt1_mask("Substructure area")
    isbv = _opt1_mask("Substructure volume")
    ispa = _opt1_mask("Superstructure area")
    ispv = _opt1_mask("Superstructure volume")
    ipow = _opt1_mask("Plant power")
    ic   = _opt1_mask("Constant")

    # ═════════════════════════════════════════════════════════════════════════
    #  Account 21 — Structures & Improvements
    # ═════════════════════════════════════════════════════════════════════════

    total_thermal_power = _get_pc(
        plant_characteristics, "Total Plant Thermal Power (MWt)",
        default=0.0, label="Account 21",
    )

    for account in df21.index.unique():
        aux    = df21.loc[account]
        method = str(aux.get("Method", ""))
        print(f"\tAccount: {account}, Name: {aux.get('Name', '?')}")

        if method == "Detailed (EEDB based)":
            # str.match returns NaN for any NaN-valued index entry.
            # _clean_mask converts those NaN results to False.
            idx = _clean_mask(
                scaling_table.index.str.match(account), scaling_table.index
            )
            print(f"\tMethod: {method}")

            portions = aux.get("Portions", None)
            subArea, subVol, superArea, superVol, bv = eval_bldg(portions, aux)

            # Use _is_nan() — handles None, float NaN, numpy scalar NaN,
            # and 0-d numpy arrays which all appear for blank Excel cells.
            inside_val = aux.get("Inside?", None)
            if not _is_nan(inside_val) and str(inside_val) != "None":
                inside_str = str(inside_val)
                if "A." in inside_str and ":" in inside_str:
                    try:
                        inside_acct = "A." + inside_str.split("A.")[1]
                        in_or_out   = inside_str.split(":")[0]
                        if in_or_out == "Inside":
                            inside_dict[inside_acct] = [
                                account, subArea, subVol, superArea, superVol, bv,
                            ]
                        elif in_or_out == "Outside":
                            if account in inside_dict:
                                (_, in_sub_a, in_sub_v,
                                 _isa, _isv, in_bv) = inside_dict[account]
                                subArea -= in_sub_a
                                subVol  -= in_sub_v
                                bv      -= in_bv
                            else:
                                print(f"\t  [WARN] 'Outside' ref for {account!r} "
                                      f"has no matching 'Inside' entry")
                    except Exception as exc:
                        print(f"\t  [WARN] Cannot parse Inside? {inside_val!r}: {exc}")
                else:
                    print(f"\t  [WARN] Unexpected Inside? format {inside_val!r}")

            print(f"\t\tSuperstructure volume: {superVol:.0f}, area: {superArea:.0f}")
            print(f"\t\tSubstructure  volume: {subVol:.0f},  area: {subArea:.0f}")
            print(f"\t\tBuilding volume: {bv:.0f}")
            plant_characteristics[account] = subArea

            # idx and ibv/isba/... are all clean bool Series — & is safe.
            _safe_loc_set(scaling_table, idx & ibv,  "New Base Unit Value", bv,                 account)
            _safe_loc_set(scaling_table, idx & isba, "New Base Unit Value", subArea,             account)
            _safe_loc_set(scaling_table, idx & isbv, "New Base Unit Value", subVol,              account)
            _safe_loc_set(scaling_table, idx & ispa, "New Base Unit Value", superArea,           account)
            _safe_loc_set(scaling_table, idx & ispv, "New Base Unit Value", superVol,            account)
            _safe_loc_set(scaling_table, idx & ic,   "New Base Unit Value", 1,                   account)
            _safe_loc_set(scaling_table, idx & ipow, "New Base Unit Value", total_thermal_power, account)

            spc = aux.get("Steel plate composite", None)
            if spc == "One sided":
                plant_characteristics["SPC One sided"].append(account)
                plant_characteristics["SPC Area"].append(superArea)
            elif spc == "Two sided":
                plant_characteristics["SPC Two sided"].append(account)
                plant_characteristics["SPC Area"].append(superArea)

            rebar = aux.get("High strength rebar", None)
            if rebar == "Grade 80":
                plant_characteristics["Grade 80"].append(account)
            elif rebar == "Grade 100":
                plant_characteristics["Grade 100"].append(account)

            if aux.get("Seismic Class 1", False):
                plant_characteristics["sc1_BV"]       += bv
                plant_characteristics["sc1_concrete"] += subVol + superVol
            if account in bv_accounts_225:
                bv_225 += bv
            concrete += subVol + superVol

            if aux.get("Name", "") == "Containment Liner":
                sup_type = aux.get("Superstructure type", "")
                n_rx = _get_pc(plant_characteristics, "Number of Reactors",
                               default=1, label="Containment liner")

                if sup_type in ("Stainless steel vessel", "Carbon steel vessel"):
                    mass = 8000.0 * (superVol + subVol)
                    print(f"\t\tMass of containment vessel: {mass:.0f}")
                    if account in scaling_table.index:
                        scaling_table.loc[account, "Option"]             = 0
                        scaling_table.loc[account, "New Base Unit Value"] = mass
                        scaling_table.loc[account, "Count per plant"]    = n_rx
                        if sup_type == "Stainless steel vessel":
                            scaling_table.loc[account, "Multipliers"] = 2.3
                    plant_characteristics["Containment type"]             = "Steel vessel"
                    plant_characteristics["Containment vessel mass (kg)"] = mass
                    plant_characteristics["Containment thickness (m)"]    = aux.get(
                        "Superstructure thickness (meters)", np.nan
                    )

                elif sup_type == "Standalone steel building":
                    plant_characteristics["Containment type"] = ["Standalone steel building"]
                    if account in scaling_table.index:
                        for mult_key, col in [
                            ("212.15 Factory cost mult",  "Factory Equipment Cost Mult"),
                            ("212.15 Labor hours mult",   "Site Labor Hours Mult"),
                            ("212.15 Labor cost mult",    "Site Labor Cost Mult"),
                            ("212.15 Material cost mult", "Site Material Cost Mult"),
                        ]:
                            if mult_key in scalars_dict:
                                scaling_table.loc[account, col] *= scalars_dict[mult_key]
                            else:
                                print(f"\t  [WARN] scalars_dict missing key {mult_key!r}")

        elif method == "Detailed (Generic)":
            print("\tDetailed (Generic) not implemented yet — skipping")

        elif method in ("Plant power scaling", "RX power scaling"):
            idx = _clean_mask(
                scaling_table.index.str.match(account), scaling_table.index
            )
            if not idx.any():
                print(f"\t  [WARN] No rows match {account!r} — skipping")
                continue
            n_rx = _get_pc(plant_characteristics, "Number of Reactors",
                           default=1, label=f"{method}/{account}") or 1
            value = (total_thermal_power if method == "Plant power scaling"
                     else _safe_divide(total_thermal_power, n_rx, f"RX/{account}"))
            _safe_loc_set(scaling_table, idx, "Option",             2,     account)
            _safe_loc_set(scaling_table, idx, "New Base Unit Value", value, account)

        elif method == "Fixed cost":
            idx = _clean_mask(
                scaling_table.index.str.match(account), scaling_table.index
            )
            if not idx.any():
                print(f"\t  [WARN] No rows match {account!r} — skipping")
                continue
            _safe_loc_set(scaling_table, idx, "Option",             4, account)
            _safe_loc_set(scaling_table, idx, "New Base Unit Value", 1, account)

        elif method == "Direct cost":
            idx = _clean_mask(
                scaling_table.index.str.match(account), scaling_table.index
            )
            if not idx.any():
                print(f"\t  [WARN] No rows match {account!r} — skipping")
                continue
            col = "Direct cost per RX (2018 USD)"
            if col in df21.columns and account in df21.index:
                _safe_loc_set(scaling_table, idx, "Option",             3,                    account)
                _safe_loc_set(scaling_table, idx, "New Base Unit Value", df21.loc[account, col], account)
            else:
                print(f"\t  [WARN] Column {col!r} or account {account!r} missing in df21")

        elif method == "":
            pass  # blank row

        else:
            print(f"\t  [WARN] Unknown method {method!r} for {account!r} — skipping")
    # ═════════════════════════════════════════════════════════════════════════
    #  Accounts 22–26
    # ═════════════════════════════════════════════════════════════════════════
    print("\nEvaluating accounts 22-26")

    # pd.concat replaces the deprecated DataFrame.append()
    df_big = pd.concat([df22, df23, df24, df25, df26])
    if "Method" in df_big.columns:
        df_big["Method"] = df_big["Method"].fillna("").astype(str)

    total_thermal = _get_pc(plant_characteristics,
                            "Total Plant Thermal Power (MWt)", default=0.0)
    net_electric  = _get_pc(plant_characteristics,
                            "Net Electrical Power (MWe)",      default=0.0)
    n_turbines    = _get_pc(plant_characteristics,
                            "Number of turbines",              default=1) or 1
    n_rx          = _get_pc(plant_characteristics,
                            "Number of Reactors",              default=1) or 1

    def _apply_method(method_name: str, option: int, value, count=None):
        """
        Apply option + value to all df_big rows matching method_name.
        NaN-indexed rows were already removed, so isin() is safe.
        """
        idx = df_big.index[df_big["Method"] == method_name]
        if idx.empty:
            return
        valid = idx[idx.isin(scaling_table.index)]
        missing = idx.difference(valid)
        if not missing.empty:
            print(f"  [WARN] {method_name}: not in scaling_table: {missing.tolist()}")
        if valid.empty:
            return
        scaling_table.loc[valid, "Option"]             = option
        scaling_table.loc[valid, "New Base Unit Value"] = value
        if count is not None:
            scaling_table.loc[valid, "Count per plant"] = count

    _apply_method("Plant power scaling",            2, total_thermal)
    _apply_method("Plant electric power scaling",   2, net_electric)
    _apply_method("Turbine electric power scaling", 2,
                  _safe_divide(net_electric, n_turbines, "Turbine EPS"),
                  count=n_turbines)
    _apply_method("RX power scaling",              2,
                  _safe_divide(total_thermal, n_rx, "RX power"),
                  count=n_rx)
    _apply_method("Fixed cost", 4, 1)

    def _apply_detailed(method_name: str, option: int,
                        value_col, count_col, fixed_value=None):
        idx = df_big.index[df_big["Method"] == method_name]
        valid = idx[idx.isin(scaling_table.index)]
        if valid.empty:
            return
        scaling_table.loc[valid, "Option"] = option
        if fixed_value is not None:
            scaling_table.loc[valid, "New Base Unit Value"] = fixed_value
        elif value_col and value_col in df_big.columns:
            scaling_table.loc[valid, "New Base Unit Value"] = df_big.loc[valid, value_col]
        if count_col and count_col in df_big.columns:
            scaling_table.loc[valid, "Count per plant"] = df_big.loc[valid, count_col]

    _apply_detailed("Detailed",          1, "Value",                          "Count per plant (DI)")
    _apply_detailed("Detailed volume",   1,  None,                             None, fixed_value=bv_225)
    _apply_detailed("Detailed pool",     0, "Value",                          "Count per plant (DI)")
    _apply_detailed("Detailed (CE)",     0, "Value",                          "Count per plant (DI)")
    _apply_detailed("Direct cost input", 3, "Direct cost per RX (2018 USD)",  "Count per plant (DCI)")

    # ── Multipliers & material uncertainty ───────────────────────────────────
    scaling_table = cost_multipliers(scaling_table, scalars_dict, plant_characteristics)
    scaling_table = material_use_uncertainty(scaling_table, scalars_dict)
    # ═════════════════════════════════════════════════════════════════════════
    #  Compute Scaling Factors
    # ════════════════════════════════════
    scaling_table["Scaling Factor"] = 0.0
    acc1 = _clean_mask(scaling_table["Option"] == 1, scaling_table.index)
    acc2 = _clean_mask(scaling_table["Option"] == 2, scaling_table.index)
    acc3 = _clean_mask(scaling_table["Option"] == 3, scaling_table.index)
    acc4 = _clean_mask(scaling_table["Option"] == 4, scaling_table.index)
    acc0 = scaling_table.index[
        _clean_mask(scaling_table["Option"] == 0, scaling_table.index)
    ]

    def _power_scale(new_col: str, base_col: str, exp_col: str) -> pd.Series:
        """(new/base)^exp — base 0 becomes NaN to avoid ZeroDivisionError."""
        new_v = scaling_table[new_col]
        base  = scaling_table[base_col].replace(0, np.nan)
        exp   = scaling_table[exp_col]
        return (new_v / base) ** exp

    if acc1.any():
        sf = _power_scale("New Base Unit Value", "EEDB Base Unit Value 1", "Option 1 Exponent")
        scaling_table.loc[acc1, "Scaling Factor"] = sf.loc[acc1]

    if acc2.any():
        sf = _power_scale("New Base Unit Value", "EEDB Base Unit Value 2", "Option 2 Exponent")
        scaling_table.loc[acc2, "Scaling Factor"] = sf.loc[acc2]

    if acc3.any():
        base3 = scaling_table.loc[acc3, "EEDB Base Unit Value 3"].replace(0, np.nan)
        scaling_table.loc[acc3, "Scaling Factor"] = (
            scaling_table.loc[acc3, "New Base Unit Value"] / base3
        )

    if acc4.any():
        scaling_table.loc[acc4, "Scaling Factor"] = 1.0

    for account in acc0:
        if account not in scaling_table.index:
            print(f"  [WARN] Option-0 account {account!r} not in index — skipping")
            continue

        raw = scaling_table.loc[account, "Option 0 Formula"]
        try:
            varz = (raw if isinstance(raw, list)
                    else [x.strip()
                          for x in str(raw).replace("[","").replace("]","").split(",")])
            varz = [float(x) for x in varz]
        except (ValueError, AttributeError) as exc:
            print(f"  [WARN] Cannot parse Option 0 Formula for {account!r}: "
                  f"{raw!r} — {exc}; Scaling Factor=0")
            continue

        new_val = scaling_table.loc[account, "New Base Unit Value"]
        base3   = scaling_table.loc[account, "EEDB Base Unit Value 3"]

        if _is_nan(base3) or base3 == 0:
            print(f"  [WARN] EEDB Base Unit Value 3 is 0/NaN for {account!r} — skipping")
            continue

        if len(varz) == 4:
            scaling_table.loc[account, "Scaling Factor"] = (
                (varz[0] + varz[1] * new_val ** varz[2]) * varz[3] / base3
            )
        elif len(varz) == 1:
            scaling_table.loc[account, "Scaling Factor"] = varz[0] * new_val / base3
        else:
            print(f"  [WARN] Option 0 Formula unexpected length {len(varz)} "
                  f"for {account!r} — skipping")

    # ── Count-per-plant multiplier ────────────────────────────────────────────
    scaling_table["Scaling Factor"] *= scaling_table["Count per plant"]

    # ── Interior concrete correction (A.212.140) ──────────────────────────────
    if "A.212.140" in scaling_table.index:
        correction = scaling_table.loc["A.212.140", "Scaling Factor"] * 8000 / (1.1 ** 3)
        plant_characteristics["sc1_concrete"] += correction
        concrete += correction
    else:
        print("  [WARN] A.212.140 not in scaling_table — concrete correction skipped")

    print(f"Concrete total: {concrete:.0f}")
    print(f"SC1 Concrete:   {plant_characteristics['sc1_concrete']:.0f}")

    return scaling_table, plant_characteristics