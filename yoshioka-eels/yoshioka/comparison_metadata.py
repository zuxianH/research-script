"""Comparison-specific provenance gates, including explicit conditional use.

User-supplied values are useful evidence. They do not become independently
verified by passing a gate. Missing generation cost only blocks cost comparisons.
"""
REQUIREMENTS={
    'array_index_diagnostics':('array_identity',),
    'conditional_spectral_shape':('array_identity','analysis_energy_axis','analysis_detector','temperature','quantum_correction'),
    'calibrated_spectral_shape':('energy_axis','native_window','momentum_axes','energy_sign'),
    'absolute_intensity':('energy_axis','momentum_axes','density_or_bin_weight','incident_normalization','elastic_handling'),
    'matched_physical_model':('specimen_geometry','electron_energy','electron_potential','force_parameter_hash','temperature','quantum_correction'),
    'runtime_memory':('generation_cost','cost_hardware','matched_accuracy_scope'),
}
TRACED={'VERIFIED','DERIVED_FROM_VERIFIED'}
CONDITIONAL=TRACED|{'USER_SUPPLIED','ANALYSIS_ASSUMPTION'}


def comparison_availability(metadata):
    result={}
    for name,keys in REQUIREMENTS.items():
        accepted=CONDITIONAL if name.startswith('conditional_') else TRACED
        # Physical comparability may be conditional on explicit user settings,
        # while still requiring every field needed by that comparison.
        if name=='matched_physical_model':accepted=CONDITIONAL-{'ANALYSIS_ASSUMPTION'}
        if name=='absolute_intensity':accepted=TRACED|{'USER_SUPPLIED'}
        missing=[k for k in keys if metadata.get(k,{}).get('status') not in accepted]
        conditional=[k for k in keys if metadata.get(k,{}).get('status') in accepted-TRACED]
        status='BLOCKED_BY_METADATA' if missing else 'AVAILABLE_CONDITIONALLY' if conditional else 'AVAILABLE'
        result[name]={'status':status,'missing':missing,'conditional_on':conditional,'requirements':list(keys)}
    return result
