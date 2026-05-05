Start the app first, then run:

```bash
cd app
python3 -m unittest discover -s tests
```

The tests assume:
- local auth exists
- bearer access tokens are used
- default admin user is `admin/admin`
