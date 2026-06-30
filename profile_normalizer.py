"""
Aligns EG and PG profiles to a common set of offsets so we can do
point-by-point cut/fill comparison. Only the overlapping range is used.

Profiles are densified before interpolation so that the shaded cut/fill
regions in validation plots tightly hug the actual lines instead of
cutting corners between sparse breakpoints.
"""

import numpy as np
from scipy import interpolate


class NormalizedProfiles:
    def __init__(self, stations, existing, proposed, interval, station_range):
        self.stations = stations
        self.existing_elevations = existing
        self.proposed_elevations = proposed
        self.station_interval = interval
        self.station_range = station_range
        self.diagnostics = {}


class ProfileNormalizer:

    def __init__(self, interval=1.0, method="linear"):
        self.interval = interval
        self.method = method

    def normalize(self, existing, proposed, sta_start=None, sta_end=None):
        if len(existing.stations) < 2:
            raise ValueError(f"EG has too few points: {len(existing.stations)}")
        if len(proposed.stations) < 2:
            raise ValueError(f"PG has too few points: {len(proposed.stations)}")

        # Densify both profiles so no gap exceeds the grid interval.
        # This is the key step: a PG line with only ~5 breakpoints will get
        # hundreds of intermediate points, ensuring fill_between() shading
        # hugs the actual road-surface lines.
        eg_sta_d, eg_elev_d = self._densify(existing.stations, existing.elevations)
        pg_sta_d, pg_elev_d = self._densify(proposed.stations, proposed.elevations)

        # overlap range -- only compute earthwork where both profiles exist
        eg_min, eg_max = float(eg_sta_d.min()), float(eg_sta_d.max())
        pg_min, pg_max = float(pg_sta_d.min()), float(pg_sta_d.max())

        if sta_start is None:
            sta_start = max(eg_min, pg_min)
        if sta_end is None:
            sta_end = min(eg_max, pg_max)

        if sta_end <= sta_start:
            raise ValueError(
                f"Profiles don't overlap -- EG [{existing.station_range}], "
                f"PG [{proposed.station_range}]"
            )

        stations = np.arange(sta_start, sta_end + self.interval / 2, self.interval)

        # Include all unique raw/densified breakpoints within the overlap range to ensure
        # the common stations grid contains the exact vertices of both profiles. This makes
        # the plotted lines and shaded boundaries hug the actual road surface perfectly
        # without cutting corners at the breakpoints.
        eg_bps = eg_sta_d[(eg_sta_d >= sta_start) & (eg_sta_d <= sta_end)]
        pg_bps = pg_sta_d[(pg_sta_d >= sta_start) & (pg_sta_d <= sta_end)]
        stations = np.unique(np.concatenate([stations, eg_bps, pg_bps]))

        ex_interp = self._interp(eg_sta_d, eg_elev_d, stations)
        pr_interp = self._interp(pg_sta_d, pg_elev_d, stations)

        result = NormalizedProfiles(stations, ex_interp, pr_interp, self.interval,
                                   (float(sta_start), float(sta_end)))
        result.diagnostics = {
            "existing_points_raw": len(existing.stations),
            "proposed_points_raw": len(proposed.stations),
            "existing_points_densified": len(eg_sta_d),
            "proposed_points_densified": len(pg_sta_d),
            "existing_range": (eg_min, eg_max),
            "proposed_range": (pg_min, pg_max),
            "overlap_range": (sta_start, sta_end),
            "overlap_width": sta_end - sta_start,
            "common_stations": len(stations),
            "method": self.method,
            "interval": self.interval,
        }
        return result

    # ------------------------------------------------------------------
    # Densification
    # ------------------------------------------------------------------

    def _densify(self, x, y):
        """Insert intermediate points so no consecutive gap exceeds *interval*.

        For each pair of consecutive breakpoints whose horizontal distance
        exceeds ``self.interval``, linearly interpolate new points at
        ``interval``-wide steps.  The original breakpoints are always
        preserved so the polyline shape is unchanged.
        """
        valid = ~(np.isnan(x) | np.isnan(y))
        xc, yc = x[valid], y[valid]

        if len(xc) < 2:
            return xc, yc

        new_x = [xc[0]]
        new_y = [yc[0]]

        for i in range(1, len(xc)):
            dx = xc[i] - xc[i - 1]
            if abs(dx) > self.interval:
                # number of sub-segments to insert
                n_seg = int(np.ceil(abs(dx) / self.interval))
                for k in range(1, n_seg):
                    t = k / n_seg
                    new_x.append(xc[i - 1] + t * (xc[i] - xc[i - 1]))
                    new_y.append(yc[i - 1] + t * (yc[i] - yc[i - 1]))
            # always append the original breakpoint
            new_x.append(xc[i])
            new_y.append(yc[i])

        return np.asarray(new_x), np.asarray(new_y)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _interp(self, x, y, x_new):
        """Interpolate within known range, clamp to edge values outside."""
        valid = ~(np.isnan(x) | np.isnan(y))
        xc, yc = x[valid], y[valid]

        if len(xc) < 2:
            raise ValueError(f"Not enough valid points for interpolation: {len(xc)}")

        if self.method == "cubic" and len(xc) >= 4:
            spline = interpolate.CubicSpline(xc, yc, bc_type="natural", extrapolate=False)
            y_new = spline(x_new)
            y_new[x_new < xc.min()] = yc[0]
            y_new[x_new > xc.max()] = yc[-1]
        else:
            y_new = np.interp(x_new, xc, yc)

        return y_new