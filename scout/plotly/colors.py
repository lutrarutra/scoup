import plotly.express as px

import scanpy as sc

PLOTLY_DISCRETE_COLORS = px.colors.qualitative.Plotly
SC_DEFAULT_COLORS = sc.pl.palettes.default_20
NONE_COLOR = "#d3d3d3"


class CaseInsensitiveDict(dict):
    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, str) else key

    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_keys()

    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(self.__class__._k(key))

    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(self.__class__._k(key), value)

    def __delitem__(self, key):
        return super(CaseInsensitiveDict, self).__delitem__(self.__class__._k(key))

    def __contains__(self, key):
        return super(CaseInsensitiveDict, self).__contains__(self.__class__._k(key))

    def has_key(self, key):
        return super(CaseInsensitiveDict, self).has_key(self.__class__._k(key))

    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).pop(self.__class__._k(key), *args, **kwargs)

    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).get(self.__class__._k(key), *args, **kwargs)

    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).setdefault(self.__class__._k(key), *args, **kwargs)

    def update(self, E={}, **F):
        super(CaseInsensitiveDict, self).update(self.__class__(E))
        super(CaseInsensitiveDict, self).update(self.__class__(**F))

    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDict, self).pop(k)
            self.__setitem__(k, v)



def get_continuous_colorscales():
    scales = CaseInsensitiveDict(dict([(clr.capitalize(), clr) for clr in px.colors.named_colorscales()]))
    scales["Seismic"] = "seismic"
    return scales

    
def get_discrete_colorscales():
    scales = {
        "ScanPy Default": SC_DEFAULT_COLORS,
        "Plotly Qualitative": PLOTLY_DISCRETE_COLORS,
    }
    for obj in dir(px.colors.qualitative):
        if not obj.startswith("_"):
            if type(getattr(px.colors.qualitative, obj)) == list:
                scales[obj.capitalize()] = getattr(px.colors.qualitative, obj)

    return CaseInsensitiveDict(scales)




def seismic(zcenter, wcenter=0.01):
    return [
        (0, "#00004C"),
        (zcenter * 0.5, "#0000E6"),
        (zcenter - zcenter * wcenter, "white"),
        (zcenter, "white"),
        (zcenter + zcenter * wcenter, "white"),
        (1 - (zcenter * 0.5), "#FF0808"),
        (1, "#840000"),
    ]