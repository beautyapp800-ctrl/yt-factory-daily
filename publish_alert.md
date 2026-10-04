**A video that should be public is not** (04.10.2026 17:27 UTC)

- **10 Stoic Lessons to Use Money as a Tool, Not a Status Symbol** - https://youtu.be/59UpZ2KJp5U
  - YouTube was asked to publish it at 2026-10-02T20:41:48+00:00
  - not public (oEmbed answered HTTP 403)
- **10 Stoic Lessons To Keep Personal Promises When No One Watches** - https://youtu.be/gRePZohxtuM
  - YouTube was asked to publish it at 2026-10-07T18:00:00Z
  - not public (oEmbed answered HTTP 403)

The usual cause is the API project not having passed audit: videos uploaded by an unaudited project are held private whatever is asked, and a schedule on one is dropped without an error. Until the audit comes through, make it public by hand in YouTube Studio: Content, open the video, Visibility.

Until it is public or the row is removed, this check will say so again every evening.
