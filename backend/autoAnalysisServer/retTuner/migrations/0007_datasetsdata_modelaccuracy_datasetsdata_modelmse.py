# Generated by Django 4.2.7 on 2024-07-05 14:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('retTuner', '0006_alter_datasetsdata_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='datasetsdata',
            name='modelaccuracy',
            field=models.FloatField(blank=True,null=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='datasetsdata',
            name='modelmse',
            field=models.FloatField(blank=True,null=True),
            preserve_default=False,
        ),
    ]
