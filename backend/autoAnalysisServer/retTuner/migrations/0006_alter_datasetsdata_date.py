# Generated by Django 4.2.7 on 2024-07-05 10:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('retTuner', '0005_rename_organization_datasetsdata_modelname_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datasetsdata',
            name='date',
            field=models.CharField(max_length=100),
        ),
    ]
