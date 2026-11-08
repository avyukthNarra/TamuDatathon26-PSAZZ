Vendored unchanged from branch `avyukth/v7c-size-flip` (commit 9481d4c, Avyukth Narra): the
`slot_forecasting/` package shipped in its built entry, renamed to `v7c_forecasting/` so it can
sit next to our own `slot_forecasting/` without a name clash (its modules use relative imports).
Used by `slot_forecasting/combo.py` for its payout-size rule and its win chance.
